# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi He and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import OrderedDict
from nets.network import Network
from model.config import cfg
import sys
import utils.timer
import matplotlib.pyplot as plt
import torch
from torchvision.ops import RoIAlign, RoIPool
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo

import torchvision
from torchvision.models.resnet import BasicBlock, Bottleneck



class ResNet(torchvision.models.resnet.ResNet):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__(block, layers, num_classes)
        # change to match the caffe resnet
        for i in range(2, 4):
            getattr(self, 'layer%d' % i)[0].conv1.stride = (2, 2)
            getattr(self, 'layer%d' % i)[0].conv2.stride = (1, 1)
        # use stride 1 for the last conv4 layer (same as tf-faster-rcnn)
        self.layer4[0].conv2.stride = (1, 1)
        self.layer4[0].downsample[0].stride = (1, 1)

        del self.avgpool, self.fc


def resnet18(pretrained=False):
    """Constructs a ResNet-18 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False):
    """Constructs a ResNet-34 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
    model = ResNet(BasicBlock, [3, 4, 6, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False):
    """Constructs a ResNet-50 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False):
    """Constructs a ResNet-101 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
                               #Blocks per layer
    model = ResNet(Bottleneck, [3, 4, 23, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False):
    """Constructs a ResNet-152 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
    model = ResNet(Bottleneck, [3, 8, 36, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model



class lidarnet(Network):
    def __init__(self, num_layers=50):
        Network.__init__(self)
        self._num_layers = num_layers
        if(cfg.LIDAR.USE_FPN):
            self._feat_stride       = 4
            self._net_conv_channels = 1024
            self._fc7_channels      = 2048
            self.inplanes = 64
        else:
            self._feat_stride       = 16
            self._net_conv_channels = 1024
            self._fc7_channels = 2048
            self.inplanes = 64
        self._roi_pooling_channels = 1024
        self.num_lidar_channels = cfg.LIDAR.NUM_CHANNEL

    def _init_modules(self):
        self._init_head_tail()

        # rpn
        self.rpn_net = nn.Conv2d(
            self._net_conv_channels, cfg.RPN_CHANNELS, [3, 3], padding=1)
        self.rpn_cls_score_net = nn.Conv2d(cfg.RPN_CHANNELS, self._num_anchors * 2, [1, 1])

        self.rpn_bbox_pred_net = nn.Conv2d(cfg.RPN_CHANNELS,
                                           self._num_anchors * 4, [1, 1])
        if(cfg.ENABLE_CUSTOM_TAIL):
            self.t_fc1           = nn.Linear(self._roi_pooling_channels,self._fc7_channels*8)
            self.t_fc2           = nn.Linear(self._fc7_channels*8,self._fc7_channels*4)
            self.t_fc3           = nn.Linear(self._fc7_channels*4,self._fc7_channels*2)
        self.cls_score_net       = nn.Linear(int(self._fc7_channels), self._num_classes)
        self.bbox_pred_net       = nn.Linear(int(self._fc7_channels), self._num_classes * 4)
        self.bbox_z_pred_net     = nn.Linear(int(self._fc7_channels), self._num_classes * 2)
        self.heading_pred_net    = nn.Linear(int(self._fc7_channels), self._num_classes)
        if(cfg.ENABLE_EPISTEMIC_BBOX_VAR):
            self.bbox_fc1        = nn.Linear(self._fc7_channels, self._fc7_channels)
            self.bbox_fc2        = nn.Linear(self._fc7_channels, int(self._fc7_channels/2))
            self.bbox_fc3        = nn.Linear(int(self._fc7_channels/2), int(self._fc7_channels/4))
            #self.bbox_dropout         = nn.Dropout(0.4)
            #self.bbox_post_dropout_fc = nn.Linear(self._fc7_channels*2, self._fc7_channels)
        if(cfg.ENABLE_ALEATORIC_BBOX_VAR):
            self.bbox_al_var_net  = nn.Linear(int(self._fc7_channels), self._num_classes * 4)
        if(cfg.ENABLE_EPISTEMIC_CLS_VAR):
            self.cls_fc1        = nn.Linear(self._fc7_channels, self._fc7_channels)
            self.cls_fc2        = nn.Linear(self._fc7_channels, int(self._fc7_channels/2))
            self.cls_fc3        = nn.Linear(int(self._fc7_channels/2), int(self._fc7_channels/4))
            #self.cls_dropout         = nn.Dropout(0.4)
            #self.cls_post_dropout_fc = nn.Linear(self._fc7_channels*2, self._fc7_channels)
        if(cfg.ENABLE_ALEATORIC_CLS_VAR):
            self.cls_al_var_net   = nn.Linear(int(self._fc7_channels/4),self._num_classes)
        self.init_weights()

    #FYI this is a fancy way of instantiating a class and calling its main function
    def _roi_pool_layer(self, bottom, rois):
        #Has restriction on batch, only one dim allowed
        return RoIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE),
                       1.0 / 4.0)(bottom, rois)

    #torchvision.ops.RoIAlign(output_size, spatial_scale, sampling_ratio)
    def _roi_align_layer(self, bottom, rois):
        return RoIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0 / 4.0,
                        0)(bottom, rois)

    def _crop_pool_layer(self, bottom, rois):
        return Network._crop_pool_layer(self, bottom, rois,
                                        cfg.RESNET.MAX_POOL)

    def _input_to_head(self,voxel_grid):
        if(cfg.LIDAR.USE_FPN):
            c1 = self._layers['img_head'](voxel_grid)
            c2 = self._layers['c2_head'](c1)
            c3 = self._layers['c3_head'](c2)
            c4 = self._layers['c4_head'](c3)
            net_conv = self._layers['fpn'](c2, c3, c4)
        else:   
            net_conv = self._layers['head'](voxel_grid)

        self._act_summaries['conv'] = net_conv
        return net_conv

    def _head_to_tail(self, pool5, dropout_en):
        #pool5 = pool5.unsqueeze(0).repeat(self._num_mc_run,1,1,1,1)
        #Reshape due to limitation on nn.conv2d (only one dim can be batch)
        #pool5 = pool5.view(-1,pool5.shape[2],pool5.shape[3],pool5.shape[4])
        fc7 = self.resnet.layer4(pool5).mean(3).mean(
            2)  # average pooling after layer4
        return fc7

    def init_weights(self):
        def uniform_init(m, min_v, max_v,bias=0.0):
            """
      weight initalizer: truncated normal and random normal.
      """
            m.weight.data.uniform_(min_v, max_v)
            #m.bias.data.zero_()
            m.bias.data.fill_(bias)

        def normal_init(m, mean, stddev, truncated=False,bias=0.0):
            """
      weight initalizer: truncated normal and random normal.
      """
            # x is a parameter
            if truncated:
                #In-place functions to save GPU mem
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(
                    mean)  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
            #m.bias.data.zero_()
            m.bias.data.fill_(bias)

        def xaiver_init(m, mean, stddev, truncated=False,bias=0.0):
            """
      weight initalizer: truncated normal and random normal.
      """
            # x is a parameter
            if truncated:
                #In-place functions to save GPU mem
                m.weight.data.xavier_normal_().fmod_(2).mul_(stddev).add_(
                    mean)  # not a perfect approximation
            else:
                nn.init.xavier_normal_(m.weight)
            #m.bias.data.zero_()
            m.bias.data.fill_(bias)
            
        
        normal_init(self.rpn_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.fpn_block.latlayer2, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.fpn_block.latlayer3, 0, 0.01, cfg.TRAIN.TRUNCATED)
        if(cfg.ENABLE_CUSTOM_TAIL):
            normal_init(self.t_fc1, 0, 0.01, cfg.TRAIN.TRUNCATED)
            normal_init(self.t_fc2, 0, 0.01, cfg.TRAIN.TRUNCATED)
            normal_init(self.t_fc3, 0, 0.01, cfg.TRAIN.TRUNCATED)

        normal_init(self.rpn_cls_score_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.rpn_bbox_pred_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.cls_score_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.bbox_pred_net, 0, 0.001, cfg.TRAIN.TRUNCATED)
        normal_init(self.bbox_z_pred_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.heading_pred_net, 0, 0.001, cfg.TRAIN.TRUNCATED)
        if(cfg.ENABLE_EPISTEMIC_BBOX_VAR):
            normal_init(self.bbox_fc1, 0, 0.01, cfg.TRAIN.TRUNCATED)
            normal_init(self.bbox_fc2, 0, 0.01, cfg.TRAIN.TRUNCATED)
            normal_init(self.bbox_fc3, 0, 0.01, cfg.TRAIN.TRUNCATED)
        if(cfg.ENABLE_EPISTEMIC_CLS_VAR):
            normal_init(self.cls_fc1, 0, 0.01, cfg.TRAIN.TRUNCATED)
            normal_init(self.cls_fc2, 0, 0.01, cfg.TRAIN.TRUNCATED)
            normal_init(self.cls_fc3, 0, 0.01, cfg.TRAIN.TRUNCATED)
        if(cfg.ENABLE_ALEATORIC_BBOX_VAR):
            normal_init(self.bbox_al_var_net, 0, 0.05, cfg.TRAIN.TRUNCATED)
        if(cfg.ENABLE_ALEATORIC_CLS_VAR):
            normal_init(self.cls_al_var_net, 0, 0.04,cfg.TRAIN.TRUNCATED)

    def _init_head_tail(self):
        # choose different blocks for different number of layers
        if self._num_layers == 50:
            self.resnet = resnet50()

        elif self._num_layers == 34:
            self.resnet = resnet34()
            
        elif self._num_layers == 101:
            self.resnet = resnet101()

        elif self._num_layers == 152:
            self.resnet = resnet152()

        else:
            # other numbers are not supported
            raise NotImplementedError

        self.resnet.conv1 = nn.Conv2d(self.num_lidar_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        # Fix blocks
        for p in self.resnet.bn1.parameters():
            p.requires_grad = False
        for p in self.resnet.conv1.parameters():
            p.requires_grad = False
        assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
        if cfg.RESNET.FIXED_BLOCKS >= 3 and cfg.PRELOAD:
            for p in self.resnet.layer3.parameters():
                p.requires_grad = False
        if cfg.RESNET.FIXED_BLOCKS >= 2 and cfg.PRELOAD:
            for p in self.resnet.layer2.parameters():
                p.requires_grad = False
        if cfg.RESNET.FIXED_BLOCKS >= 1 and cfg.PRELOAD:
            for p in self.resnet.layer1.parameters():
                p.requires_grad = False

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters():
                    p.requires_grad = False

        self.resnet.apply(set_bn_fix)
        if(cfg.LIDAR.USE_FPN):
            self.fpn_block = BuildBlock()
            self._layers['fpn'] = self.fpn_block
        # Build resnet.
        self._layers['img_head'] = nn.Sequential(
            self.resnet.conv1, self.resnet.bn1, self.resnet.relu,self.resnet.maxpool)
        self._layers['c2_head'] = self.resnet.layer1
        self._layers['c3_head'] = self.resnet.layer2
        self._layers['c4_head'] = self.resnet.layer3

    def _region_classification(self, fc7):
        #fc7 = fc7.unsqueeze(0).repeat(self._num_mc_run,1,1)
        if(cfg.ENABLE_EPISTEMIC_CLS_VAR):
            cls_score_in = self._cls_tail(fc7,True)
        else:
            cls_score_in = fc7
        cls_score = self.cls_score_net(cls_score_in)
        if(cfg.ENABLE_EPISTEMIC_BBOX_VAR):
            bbox_pred_in = self._bbox_tail(fc7,True)
        else:
            bbox_pred_in = fc7
        bbox_s_pred  = self.bbox_pred_net(bbox_pred_in)
        num_samp = bbox_s_pred.shape[0]
        heading_pred = self.heading_pred_net(bbox_pred_in)
        #heading_pred = torch.tanh(heading_pred)
        bbox_z_pred  = self.bbox_z_pred_net(bbox_pred_in)
        #Mix heading back into bbox pred
        bbox_pred    = torch.cat((bbox_s_pred.view(num_samp,-1,4),bbox_z_pred.view(num_samp,-1,2),heading_pred.view(num_samp,-1,1)),dim=2)
        bbox_pred    = bbox_pred.view(num_samp,-1,7*self._num_classes)
        cls_score_mean = torch.mean(cls_score,dim=0)
        cls_pred = torch.max(cls_score_mean, 1)[1]
        cls_prob = torch.mean(F.softmax(cls_score, dim=2),dim=0)
        self._mc_run_output['bbox_pred'] = bbox_pred
        self._mc_run_output['cls_score'] = cls_score
        self._predictions['cls_score'] = cls_score_mean
        self._predictions['cls_pred'] = cls_pred
        self._predictions['cls_prob'] = cls_prob
        #TODO: Make domain shift here
        self._predictions['bbox_pred'] = torch.mean(bbox_pred,dim=0)
        if(cfg.ENABLE_ALEATORIC_BBOX_VAR):
            bbox_var  = self.bbox_al_var_net(fc7)
            self._predictions['a_bbox_var']  = torch.mean(bbox_var,dim=0)
        if(cfg.ENABLE_ALEATORIC_CLS_VAR):
            a_cls_var   = self.cls_al_var_net(cls_score_in)
            a_cls_var = torch.exp(torch.mean(a_cls_var,dim=0))
            self._predictions['a_cls_var']   = a_cls_var

    def _cls_tail(self,fc7,dropout_en):
        if(dropout_en):
            fc_dropout_rate   = 0.5
        else:
            fc_dropout_rate   = 0.0
        fc_dropout1   = nn.Dropout(fc_dropout_rate)
        fc_dropout2   = nn.Dropout(fc_dropout_rate)
        fc_dropout3   = nn.Dropout(fc_dropout_rate)
        fc_relu      = nn.ReLU(inplace=True)
        fc1     = self.cls_fc1(fc7)
        fc1_r   = fc_relu(fc1)
        fc1_d   = fc_dropout1(fc1_r)
        fc2     = self.cls_fc2(fc1_d)
        fc2_r   = fc_relu(fc2)
        fc2_d   = fc_dropout2(fc2_r)
        fc3     = self.cls_fc3(fc2_d)
        fc3_r   = fc_relu(fc3)
        fc3_d   = fc_dropout2(fc3_r)
        return fc3_d

    def _bbox_tail(self,fc7,dropout_en):
        if(dropout_en):
            fc_dropout_rate   = 0.4
        else:
            fc_dropout_rate   = 0.0
        fc_dropout1   = nn.Dropout(fc_dropout_rate)
        fc_dropout2   = nn.Dropout(fc_dropout_rate)
        fc_dropout3   = nn.Dropout(fc_dropout_rate)
        fc_relu      = nn.ReLU(inplace=True)
        fc1     = self.bbox_fc1(fc7)
        fc1_r   = fc_relu(fc1)
        fc1_d   = fc_dropout1(fc1_r)
        fc2     = self.bbox_fc2(fc1_d)
        fc2_r   = fc_relu(fc2)
        fc2_d   = fc_dropout2(fc2_r)
        fc3     = self.bbox_fc3(fc2_d)
        fc3_r   = fc_relu(fc3)
        fc3_d   = fc_dropout2(fc3_r)
        return fc2_d

    def _custom_tail(self,pool5,dropout_en):
        pool5 = pool5.mean(3).mean(2).unsqueeze(0).repeat(self._num_mc_run,1,1)
        if(dropout_en):
            conv_dropout_rate = 0.2
            fc_dropout_rate   = 0.5
        else:
            conv_dropout_rate = 0.0
            fc_dropout_rate   = 0.0
        pool_dropout = nn.Dropout(conv_dropout_rate)
        fc_dropout1   = nn.Dropout(fc_dropout_rate)
        fc_dropout2   = nn.Dropout(fc_dropout_rate)
        fc_dropout3   = nn.Dropout(fc_dropout_rate)
        fc_dropout4   = nn.Dropout(fc_dropout_rate)
        fc_relu      = nn.ReLU(inplace=True)
        pool5_d = pool_dropout(pool5)
        fc1     = self.t_fc1(pool5_d)
        fc1_r   = fc_relu(fc1)
        fc1_d   = fc_dropout1(fc1_r)
        fc2     = self.t_fc2(fc1_d)
        fc2_r   = fc_relu(fc2)
        fc2_d   = fc_dropout2(fc2_r)
        fc3     = self.t_fc3(fc2_d)
        fc3_r   = fc_relu(fc3)
        fc3_d   = fc_dropout3(fc3_r)
        return fc3_d


    def train(self, mode=True):
        # Override train so that the training mode is set as we want
        nn.Module.train(self, mode)
        if mode:
            # Set fixed blocks to be in eval mode (not really doing anything)
            self.resnet.eval()
            if cfg.RESNET.FIXED_BLOCKS <= 3:
                self.resnet.layer4.train()
            if cfg.RESNET.FIXED_BLOCKS <= 2:
                self.resnet.layer3.train()
            if cfg.RESNET.FIXED_BLOCKS <= 1:
                self.resnet.layer2.train()
            if cfg.RESNET.FIXED_BLOCKS == 0:
                self.resnet.layer1.train()

            # Set batchnorm always in eval mode during training
            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self.resnet.apply(set_bn_eval)
            
    def key_transform(self, key):
        #if('resnet.' in key):
        #    new_key = key.replace('resnet.', '')
        #    return new_key
        if('RCNN_base' in key):
            new_key = key.replace('RCNN_base.4.', 'layer1.').replace('RCNN_base.5.', 'layer2.').replace('RCNN_base.6.', 'layer3.').replace('RCNN_base.1.', 'bn1.').replace('RCNN_base.0.','conv1.')
            return new_key
        elif('RCNN_top' in key):
            new_key = key.replace('RCNN_top.0.', 'layer4.')
            return new_key
        else:
            return None


    def load_pretrained_rpn(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)


    def load_imagenet_pretrained_cnn(self, state_dict):
        new_state_dict = OrderedDict()
        own_state = self.state_dict()
        #print(state_dict.keys())
        #sys.exit('test_ended')
        resnet_state_dict = state_dict
        for key, param in resnet_state_dict.items():
            #new_key = self.key_transform(key)
            if(key == 'conv1.weight'):
                new_param = torch.zeros((param.shape[0],cfg.LIDAR.NUM_CHANNEL,param.shape[2],param.shape[3]))
                r = param[:,0,:,:].unsqueeze(1)
                new_param[:,0:cfg.LIDAR.NUM_SLICES,:,:] = r.repeat(1,cfg.LIDAR.NUM_SLICES,1,1)
                new_param[:,cfg.LIDAR.NUM_SLICES,:,:] = param[:,1,:,:]
                new_param[:,cfg.LIDAR.NUM_SLICES+1,:,:] = param[:,2,:,:]
            else:
                new_param = param
            #if(new_key is not None):
            new_state_dict[key] = new_param
        self._load_pretrained_cnn(new_state_dict)

    def load_pretrained_cnn(self, state_dict):
        sd = {}
        for k, v in state_dict.items():
            if 'resnet' in k:
                k = k.replace('resnet.','')
                sd[k] = v
        self.resnet.load_state_dict(sd)

class BuildBlock(nn.Module):
  def __init__(self, planes=1024):
    super(BuildBlock, self).__init__()
    # Top-down layers, use nn.ConvTranspose2d to replace nn.Conv2d+F.upsample?
    #self.toplayer1 = nn.Conv2d(2048, planes, kernel_size=1, stride=1, padding=0)  # Reduce channels
    #self.toplayer2 = nn.Conv2d(1024, planes, kernel_size=3, stride=1, padding=1)
    #self.toplayer3 = nn.Conv2d(256, planes, kernel_size=3, stride=1, padding=1)
    #self.toplayer4 = nn.Conv2d(256, planes, kernel_size=3, stride=1, padding=1)

    # Lateral layers
    #self.latlayer1 = nn.Conv2d(1024, planes, kernel_size=1, stride=1, padding=0)
    self.latlayer2 = nn.Conv2d(512, planes, kernel_size=1, stride=1, padding=0)
    #self.latlayer2 = nn.Conv2d( 512, planes, kernel_size=1, stride=1, padding=0)
    self.latlayer3 = nn.Conv2d(256, planes, kernel_size=1, stride=1, padding=0)

    self.subsample = nn.AvgPool2d(2, stride=2)

  def _upsample_add(self, x, y):
    _,_,H,W = y.size()
    #print('fpn upsample dims H: {} W: {}'.format(H,W))
    return F.interpolate(x, size=(H,W), mode='bilinear',align_corners=False) + y

  def forward(self, c2, c3, c4):
    # Top-down
    #p5 = self.toplayer1(c5)
    #p6 = self.subsample(p5)
    #p4 = self._upsample_add(p5, self.latlayer1(c4))
    #p4 = self.toplayer2(c4)
    p3 = self._upsample_add(c4, self.latlayer2(c3))
    #p3 = self.toplayer3(p3)
    p2 = self._upsample_add(p3, self.latlayer3(c2))
    #p2 = self.toplayer4(p2)

    return p2