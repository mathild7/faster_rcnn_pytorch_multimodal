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
from nets.fpn import fpn
from utils.init_utils import xaiver_init, const_init, normal_init, uniform_init, set_bn_fix, set_bn_var, set_bn_train, set_bn_eval
import torchvision
#from torchvision.models.resnet import BasicBlock, Bottleneck
import nets.resnet as custom_resnet

class imagenet(Network):
    def __init__(self, num_layers=50):
        Network.__init__(self)
        if(cfg.USE_FPN):
            #WAS 4 now is 8 due to downsample layer after FPN
            if(cfg.POOLING_MODE == 'multiscale'):
                self._feat_stride = 4
            #DEPRECATED (see lidar for explanation)
            #else:
            #    self._feat_stride = 8
            self._fpn_en               = True
            self._batchnorm_en         = True
            self._net_conv_channels    = 256
            self._roi_pooling_channels = cfg.POOLING_SIZE*cfg.POOLING_SIZE*self._net_conv_channels
        else:
            self._feat_stride          = 16
            self._fpn_en               = False
            self._batchnorm_en         = True
            self._net_conv_channels    = 1024
            self._roi_pooling_channels = 1024
        self._fc7_channels         = 2048
        self.inplanes              = 64
        self._num_resnet_layers    = num_layers
        if(cfg.UC.EN_BBOX_EPISTEMIC or cfg.UC.EN_CLS_EPISTEMIC):
            self._det_net_channels = int(self._fc7_channels/4)
            self._dropout_en       = True
            self._cls_drop_rate    = 0.3
            self._bbox_drop_rate   = 0.1
            self._resnet_drop_rate = 0.5
        else:
            self._det_net_channels = self._fc7_channels
            self._dropout_en       = False
            self._cls_drop_rate    = 0.0
            self._bbox_drop_rate   = 0.0
            self._resnet_drop_rate = 0.0

    def init_weights(self):
        normal_init(self.rpn_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
        if(cfg.USE_FPN):
            self._fpn.init()
        #FPN custom tail
        if(cfg.ENABLE_CUSTOM_TAIL):
            normal_init(self.t_fc1, 0, 0.01, cfg.TRAIN.TRUNCATED)
            normal_init(self.t_fc2, 0, 0.01, cfg.TRAIN.TRUNCATED)
            normal_init(self.t_fc3, 0, 0.01, cfg.TRAIN.TRUNCATED)
        #Bbox epistemic uncertainty FC for dropout
        if(cfg.UC.EN_BBOX_EPISTEMIC):
            normal_init(self.bbox_fc1, 0, 0.01, cfg.TRAIN.TRUNCATED)
            normal_init(self.bbox_fc2, 0, 0.01, cfg.TRAIN.TRUNCATED)
        #Class epistemic uncertainty FC for dropout
        if(cfg.UC.EN_CLS_EPISTEMIC):
            normal_init(self.cls_fc1, 0, 0.01, cfg.TRAIN.TRUNCATED)
            normal_init(self.cls_fc2, 0, 0.01, cfg.TRAIN.TRUNCATED)
        #Detection head
        normal_init(self.rpn_cls_score_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.rpn_bbox_pred_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.cls_score_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.bbox_pred_net, 0, 0.001, cfg.TRAIN.TRUNCATED)
        #Aleatoric Uncertainty head
        if(cfg.UC.EN_BBOX_ALEATORIC):
            normal_init(self.bbox_al_var_net, 0, 0.001, True)
        if(cfg.UC.EN_CLS_ALEATORIC):
            normal_init(self.cls_al_var_net, 0, 0.01, True)

    def _init_head_tail(self):
        self.resnet = self._build_resnet()
        # Fix blocks
        for p in self.resnet.bn1.parameters():
            p.requires_grad = False
        for p in self.resnet.conv1.parameters():
            p.requires_grad = False
        assert (-1 <= cfg.RESNET.FIXED_BLOCKS < 4)
        if cfg.RESNET.FIXED_BLOCKS >= 3:
            for p in self.resnet.layer3.parameters():
                p.requires_grad = False
        if cfg.RESNET.FIXED_BLOCKS >= 2:
            for p in self.resnet.layer2.parameters():
                p.requires_grad = False
        if cfg.RESNET.FIXED_BLOCKS >= 1:
            for p in self.resnet.layer1.parameters():
                p.requires_grad = False
        if(cfg.RESNET.FIXED_BLOCKS == -1):
            self.resnet.apply(set_bn_var)
        #elif(cfg.RESNET.FIXED_BLOCKS == 1 and cfg.USE_FPN):
        #    self.resnet.apply(set_bn_fix)
        #    self.resnet.layer4.apply(set_bn_var)
        else:
            self.resnet.apply(set_bn_fix)

        # Build resnet.

        if(cfg.USE_FPN):
            self._fpn = fpn(planes=self._net_conv_channels)
            self._layers['fpn'] = self._fpn
            self._layers['fpn_downsample'] = nn.MaxPool2d(2)
            self._layers['head'] = nn.Sequential(
                self.resnet.conv1, self.resnet.bn1, self.resnet.relu,self.resnet.maxpool)
            self._layers['layer1'] = self.resnet.layer1
            self._layers['layer2'] = self.resnet.layer2
            self._layers['layer3'] = self.resnet.layer3
            self._layers['layer4'] = self.resnet.layer4
        else:
            self._layers['head'] = nn.Sequential(
                self.resnet.conv1, self.resnet.bn1, self.resnet.relu,
                self.resnet.maxpool, self.resnet.layer1, self.resnet.layer2,
                self.resnet.layer3)
        # Build resnet.


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
                self.resnet.conv1.train()

            #This goes after the case/switch above due to imagenet either having all batchnorm train(0-4) or all freeze (-1)
            if(cfg.RESNET.FIXED_BLOCKS == -1):
                self.resnet.train()
                self.resnet.apply(set_bn_train)
            #elif(cfg.RESNET.FIXED_BLOCKS == 1 and cfg.USE_FPN):
            #    self.resnet.apply(set_bn_eval)
            #    self.resnet.layer4.apply(set_bn_train)
            else:
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
        self._load_pretrained_cnn(new_state_dict)

    def load_pretrained_rpn(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)

    def load_pretrained_full(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            print(name)
            if name not in own_state:
                continue
            if 'bbox' in name and 'rpn' not in name:
                continue
            if 'cls' in name and 'rpn' not in name:
                continue
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)

    def load_pretrained_cnn(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if('resnet' not in name):
                name = 'resnet.'+name
            if name not in own_state:
                continue
            #if(cfg.USE_FPN):
            #    if 'layer4' in name and 'bn' not in name:
            #        continue
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)


    def _load_pretrained_cnn(self, state_dict):
        self.resnet.load_state_dict({
            k: v
            for k, v in state_dict.items() if k in self.resnet.state_dict()
        })
