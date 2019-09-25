# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nets.network import Network
from model.config import cfg
from collections import OrderedDict
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torchvision.models as models


class vgg16(Network):
    def __init__(self):
        Network.__init__(self)
        self._feat_stride = [
            16,
        ]
        self._feat_compress = [
            1. / float(self._feat_stride[0]),
        ]
        self._net_conv_channels = 512
        self._fc7_channels = 4096

    def _init_head_tail(self):
        self.vgg = models.vgg16()
        # Remove fc8
        self.vgg.classifier = nn.Sequential(
            *list(self.vgg.classifier._modules.values())[:-1])

        # Fix the layers before conv3:
        for layer in range(10):
            for p in self.vgg.features[layer].parameters():
                p.requires_grad = False

        # not using the last maxpool layer
        self._layers['head'] = nn.Sequential(
            *list(self.vgg.features._modules.values())[:-1])

    def _image_to_head(self):
        net_conv = self._layers['head'](self._image)
        self._act_summaries['conv'] = net_conv

        return net_conv

    def _head_to_tail(self, pool5):
        pool5_flat = pool5.view(pool5.size(0), -1)
        fc7 = self.vgg.classifier(pool5_flat)

        return fc7

    def key_transform(self, key):
        #if('resnet.' in key):
        #    new_key = key.replace('resnet.', '')
        #    return new_key
        if('features.' in key or 'classifier.' in key):
            new_key = key
            return new_key
        else:
            return None

    def load_trimmed_pretrained_cnn(self, state_dict):
        new_state_dict = OrderedDict()
        own_state = self.state_dict()
        print(state_dict.keys())
        #sys.exit('test_ended')
        for key, param in state_dict.items():
            new_key = self.key_transform(key)
            if(new_key is not None):
                new_state_dict[new_key] = param
        self.load_pretrained_cnn(new_state_dict)


    def load_pretrained_cnn(self, state_dict):
        self.vgg.load_state_dict({
            k: v
            for k, v in state_dict.items() if k in self.vgg.state_dict()
        })