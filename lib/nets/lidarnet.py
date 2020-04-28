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
#from torchvision.models.resnet import BasicBlock, Bottleneck
import nets.resnet as custom_resnet

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

        if(cfg.UC.EN_BBOX_EPISTEMIC or cfg.UC.EN_CLS_EPISTEMIC):
            self._det_net_channels = int(self._fc7_channels/4)
            self._dropout_en       = True
            self._drop_rate        = 0.2
            self._fc_drop_rate     = 0.4
        else:
            self._det_net_channels = self._fc7_channels
            self._dropout_en       = False
            self._drop_rate        = 0.0
            self._fc_drop_rate     = 0.0
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

        #Epistemic dropout layers
        if(cfg.UC.EN_BBOX_EPISTEMIC):
            self.bbox_fc1        = nn.Linear(self._fc7_channels, self._det_net_channels*2)
            self.bbox_bn1        = nn.BatchNorm1d(self._det_net_channels*2)
            self.bbox_drop1      = nn.Dropout(self._fc_drop_rate)
            self.bbox_fc2        = nn.Linear(self._det_net_channels*2, self._det_net_channels)
            self.bbox_bn2        = nn.BatchNorm1d(self._det_net_channels)
            self.bbox_drop2      = nn.Dropout(self._fc_drop_rate)
        #    self.bbox_fc3        = nn.Linear(self._det_net_channels*2, self._det_net_channels)
        if(cfg.UC.EN_CLS_EPISTEMIC):
            self.cls_fc1        = nn.Linear(self._fc7_channels, self._det_net_channels*2)
            self.cls_bn1        = nn.BatchNorm1d(self._det_net_channels*2)
            self.cls_drop1      = nn.Dropout(self._fc_drop_rate)
            self.cls_fc2        = nn.Linear(self._det_net_channels*2, self._det_net_channels)
            self.cls_bn2        = nn.BatchNorm1d(self._det_net_channels)
            self.cls_drop2      = nn.Dropout(self._fc_drop_rate)
        #    self.cls_fc3        = nn.Linear(self._det_net_channels*2, self._det_net_channels)

        #Traditional outputs
        self.cls_score_net       = nn.Linear(self._det_net_channels, self._num_classes)
        self.bbox_pred_net       = nn.Linear(self._det_net_channels, self._num_classes * cfg.IMAGE.NUM_BBOX_ELEM)
        self.bbox_z_pred_net     = nn.Linear(self._det_net_channels, self._num_classes * 2)
        self.heading_pred_net    = nn.Linear(self._det_net_channels, self._num_classes)

        #Aleatoric leafs
        if(cfg.UC.EN_CLS_ALEATORIC):
            self.cls_al_var_net   = nn.Linear(self._det_net_channels,self._num_classes)
        if(cfg.UC.EN_BBOX_ALEATORIC):
            self.bbox_al_var_net  = nn.Linear(self._det_net_channels, self._num_classes * cfg.LIDAR.NUM_BBOX_ELEM)
        self.init_weights()


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

        def const_init(m, weight, bias):
            nn.init.constant_(m.weight, weight)
            nn.init.constant_(m.bias, bias)

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
        if(cfg.UC.EN_BBOX_EPISTEMIC):
            normal_init(self.bbox_fc1, 0, 0.01, cfg.TRAIN.TRUNCATED)
            normal_init(self.bbox_fc2, 0, 0.01, cfg.TRAIN.TRUNCATED)
            const_init(self.bbox_bn1, 1.0, 0.0)
            const_init(self.bbox_bn2, 1.0, 0.0)
        #    normal_init(self.bbox_fc3, 0, 0.01, cfg.TRAIN.TRUNCATED)
        if(cfg.UC.EN_CLS_EPISTEMIC):
            normal_init(self.cls_fc1, 0, 0.01, cfg.TRAIN.TRUNCATED)
            normal_init(self.cls_fc2, 0, 0.01, cfg.TRAIN.TRUNCATED)
            const_init(self.cls_bn1, 1.0, 0.0)
            const_init(self.cls_bn2, 1.0, 0.0)
        #    normal_init(self.cls_fc3, 0, 0.01, cfg.TRAIN.TRUNCATED)
        if(cfg.UC.EN_BBOX_ALEATORIC):
            normal_init(self.bbox_al_var_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
        if(cfg.UC.EN_CLS_ALEATORIC):
            normal_init(self.cls_al_var_net, 0, 0.04, cfg.TRAIN.TRUNCATED)

    def _init_head_tail(self):
        # choose different blocks for different number of layers
        if self._num_layers == 50:
            self.resnet = custom_resnet.resnet50(dropout_en=self._dropout_en,drop_rate=self._drop_rate)

        elif self._num_layers == 34:
            self.resnet = custom_resnet.resnet34(dropout_en=self._dropout_en,drop_rate=self._drop_rate)
            
        elif self._num_layers == 101:
            self.resnet = custom_resnet.resnet101(dropout_en=self._dropout_en,drop_rate=self._drop_rate)

        elif self._num_layers == 152:
            self.resnet = custom_resnet.resnet152(dropout_en=self._dropout_en,drop_rate=self._drop_rate)

        else:
            # other numbers are not supported
            raise NotImplementedError

        self.resnet.conv1 = nn.Conv2d(self.num_lidar_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        # Fix blocks
        for p in self.resnet.bn1.parameters():
            if(cfg.RESNET.FIXED_BLOCKS >= 1):
                p.requires_grad = False
        for p in self.resnet.conv1.parameters():
            if(cfg.RESNET.FIXED_BLOCKS >= 1):
                p.requires_grad = False
        assert (-1 <= cfg.RESNET.FIXED_BLOCKS < 4)
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
        #Disable, as this has not been trained yet
        if(cfg.RESNET.FIXED_BLOCKS >= 0):
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
            if cfg.RESNET.FIXED_BLOCKS <= 0:
                self.resnet.layer1.train()
                #self.resnet.conv1.train()
            if cfg.RESNET.FIXED_BLOCKS == -1:
                self.resnet.train()

            # Set batchnorm always in eval mode during training
            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()
            #Disable as resnet has not been trained yet
            if(cfg.RESNET.FIXED_BLOCKS >= 0):
                self.resnet.apply(set_bn_eval)
            
    def eval(self):
        nn.Module.eval(self)
        if(cfg.ENABLE_FULL_NET is True and cfg.UC.EN_BBOX_EPISTEMIC):
            self.bbox_drop1.train()
            self.bbox_drop2.train()
        if(cfg.ENABLE_FULL_NET is True and cfg.UC.EN_CLS_EPISTEMIC):
            self.cls_drop1.train()
            self.cls_drop2.train()

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
            if 'resnet' not in name and 'fpn' not in name and 'rpn' not in name:
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
