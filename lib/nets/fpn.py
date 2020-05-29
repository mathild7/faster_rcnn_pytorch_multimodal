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

from utils.init_utils import xaiver_init, const_init, normal_init, uniform_init, set_bn_fix, set_bn_var
import torchvision
#from torchvision.models.resnet import BasicBlock, Bottleneck
import nets.resnet as custom_resnet

class fpn(nn.Module):
  def __init__(self, c2_inplanes=256, c3_inplanes=512, c4_inplanes=1024, c5_inplanes=2048, planes=1024):
    super().__init__()
    # Top-down layers, use nn.ConvTranspose2d to replace nn.Conv2d+F.upsample?
    #self.toplayer1 = nn.Conv2d(2048, planes, kernel_size=1, stride=1, padding=0)  # Reduce channels
    #self.toplayer2 = nn.Conv2d(1024, planes, kernel_size=3, stride=1, padding=1)
    #self.toplayer3 = nn.Conv2d(256, planes, kernel_size=3, stride=1, padding=1)
    #self.toplayer4 = nn.Conv2d(256, planes, kernel_size=3, stride=1, padding=1)

    # Lateral layers
    self.latlayer2 = nn.Conv2d(c2_inplanes, planes, kernel_size=1, stride=1, padding=0)
    self.latlayer3 = nn.Conv2d(c3_inplanes, planes, kernel_size=1, stride=1, padding=0)
    self.latlayer4 = nn.Conv2d(c4_inplanes, planes, kernel_size=1, stride=1, padding=0)
    self.latlayer5 = nn.Conv2d(c5_inplanes, planes, kernel_size=1, stride=1, padding=0)
    self.aalayer2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
    self.aalayer3 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
    self.aalayer4 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
    self.subsample = nn.AvgPool2d(2, stride=2)

  def _upsample_add(self, x, y):
    _,_,H,W = y.size()
    #print('fpn upsample dims H: {} W: {}'.format(H,W))
    return F.interpolate(x, size=(H,W), mode='bilinear',align_corners=False) + y

  def forward(self, c2, c3, c4, c5):
    # Top-down
    p5 = self.latlayer5(c5)
    p4 = self.latlayer4(c4)
    p4 = self._upsample_add(p5, p4)
    p3 = self.latlayer3(c3)
    p3 = self._upsample_add(p4, p3)
    p3 = self.aalayer3(p3)
    p2 = self.latlayer2(c2)
    p2 = self._upsample_add(p3, p2)
    p2 = self.aalayer2(p2)

    return p2, p3, p4, p5
