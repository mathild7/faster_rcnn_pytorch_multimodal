# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi He and Xinlei Chen
#THIS WILL BE USEFUL: https://github.com/yxgeee/pytorch-FPN/blob/master/lib/nets/resnet_v1.py
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import OrderedDict
from nets.network import Network
from model.config import cfg
import sys
import utils.timer
from torchvision.ops import RoIAlign, RoIPool
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo

import torchvision
#from torchvision.models.resnet import BasicBlock, Bottleneck
import nets.resnet as custom_resnet

class resnetv1(Network):
    def __init__(self, num_layers=50):
        Network.__init__(self)
        #TODO: Why is this an array, might cause a problem
        self._feat_stride = [
            16,
        ]
        self._feat_compress = [
            1. / float(self._feat_stride[0]),
        ]
        self._num_layers = num_layers
        self._net_conv_channels = 1024
        self._fc7_channels = 2048
        self._roi_pooling_channels = 1024
        if(cfg.UC.EN_BBOX_EPISTEMIC or cfg.UC.EN_CLS_EPISTEMIC):
            self._det_net_channels = int(self._fc7_channels/4)
            self._dropout_en       = True
            self._drop_rate        = 0.2
        else:
            self._det_net_channels = self._fc7_channels
            self._dropout_en       = False
            self._drop_rate        = 0.0
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
        #if(cfg.UC.EN_BBOX_EPISTEMIC):
        #    self.bbox_fc1        = nn.Linear(self._fc7_channels, self._fc7_channels)
        #    self.bbox_fc2        = nn.Linear(self._fc7_channels, int(self._fc7_channels/2))
        #    self.bbox_fc3        = nn.Linear(int(self._fc7_channels/2), self._det_net_channels)
        #if(cfg.UC.EN_CLS_EPISTEMIC):
        #    self.cls_fc1        = nn.Linear(self._fc7_channels, self._fc7_channels)
        #    self.cls_fc2        = nn.Linear(self._fc7_channels, int(self._fc7_channels/2))
        #    self.cls_fc3        = nn.Linear(int(self._fc7_channels/2), self._det_net_channels)

        #Traditional outputs
        self.cls_score_net       = nn.Linear(self._fc7_channels, self._num_classes)
        self.bbox_pred_net       = nn.Linear(self._fc7_channels, self._num_classes * cfg.IMAGE.NUM_BBOX_ELEM)

        #Aleatoric leafs
        if(cfg.UC.EN_CLS_ALEATORIC):
            self.cls_al_var_net   = nn.Linear(self._fc7_channels,self._num_classes)
        if(cfg.UC.EN_BBOX_ALEATORIC):
            self.bbox_al_var_net  = nn.Linear(self._fc7_channels, self._num_classes * cfg.IMAGE.NUM_BBOX_ELEM)
        self.init_weights()

    #FYI this is a fancy way of instantiating a class and calling its main function
    def _roi_pool_layer(self, bottom, rois):
        #Has restriction on batch, only one dim allowed
        return RoIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE),
                       1.0 / 16.0)(bottom, rois)

    def _roi_align_layer(self, bottom, rois):
        return RoIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0 / 16.0,
                        0)(bottom, rois)

    def _crop_pool_layer(self, bottom, rois):
        return Network._crop_pool_layer(self, bottom, rois,
                                        cfg.RESNET.MAX_POOL)

    def _input_to_head(self,image):
        net_conv = self._layers['head'](image)
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
        
        normal_init(self.rpn_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
        if(cfg.ENABLE_CUSTOM_TAIL):
            normal_init(self.t_fc1, 0, 0.01, cfg.TRAIN.TRUNCATED)
            normal_init(self.t_fc2, 0, 0.01, cfg.TRAIN.TRUNCATED)
            normal_init(self.t_fc3, 0, 0.01, cfg.TRAIN.TRUNCATED)

        normal_init(self.rpn_cls_score_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.rpn_bbox_pred_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.cls_score_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.bbox_pred_net, 0, 0.001, cfg.TRAIN.TRUNCATED)
        #if(cfg.UC.EN_BBOX_EPISTEMIC):
        #    normal_init(self.bbox_fc1, 0, 0.01, cfg.TRAIN.TRUNCATED)
        #    normal_init(self.bbox_fc2, 0, 0.01, cfg.TRAIN.TRUNCATED)
        #    normal_init(self.bbox_fc3, 0, 0.01, cfg.TRAIN.TRUNCATED)
        #if(cfg.UC.EN_CLS_EPISTEMIC):
        #    normal_init(self.cls_fc1, 0, 0.01, cfg.TRAIN.TRUNCATED)
        #    normal_init(self.cls_fc2, 0, 0.01, cfg.TRAIN.TRUNCATED)
        #    normal_init(self.cls_fc3, 0, 0.01, cfg.TRAIN.TRUNCATED)
        if(cfg.UC.EN_BBOX_ALEATORIC):
            normal_init(self.bbox_al_var_net, 0, 0.05, cfg.TRAIN.TRUNCATED)
        if(cfg.UC.EN_CLS_ALEATORIC):
            normal_init(self.cls_al_var_net, 0, 0.04,cfg.TRAIN.TRUNCATED)

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

        # Fix blocks
        for p in self.resnet.bn1.parameters():
            p.requires_grad = False
        for p in self.resnet.conv1.parameters():
            p.requires_grad = False
        assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
        if cfg.RESNET.FIXED_BLOCKS >= 3:
            for p in self.resnet.layer3.parameters():
                p.requires_grad = False
        if cfg.RESNET.FIXED_BLOCKS >= 2:
            for p in self.resnet.layer2.parameters():
                p.requires_grad = False
        if cfg.RESNET.FIXED_BLOCKS >= 1:
            for p in self.resnet.layer1.parameters():
                p.requires_grad = False

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters():
                    p.requires_grad = False

        self.resnet.apply(set_bn_fix)

        # Build resnet.
        self._layers['head'] = nn.Sequential(
            self.resnet.conv1, self.resnet.bn1, self.resnet.relu,
            self.resnet.maxpool, self.resnet.layer1, self.resnet.layer2,
            self.resnet.layer3)


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

    def load_trimmed_pretrained_cnn(self, state_dict):
        new_state_dict = OrderedDict()
        own_state = self.state_dict()
        print(state_dict.keys())
        sys.exit('test_ended')
        resnet_state_dict = (state_dict['model'])
        for key, param in resnet_state_dict.items():
            new_key = self.key_transform(key)
            if(new_key is not None):
                new_state_dict[new_key] = param
        self.load_pretrained_cnn(new_state_dict)

    def load_pretrained_rpn(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)

    def load_pretrained_cnn(self, state_dict):
        self.resnet.load_state_dict({
            k: v
            for k, v in state_dict.items() if k in self.resnet.state_dict()
        })
