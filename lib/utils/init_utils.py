from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

from model.config import cfg

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

def set_bn_fix(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        for p in m.parameters():
            p.requires_grad = False

def set_bn_var(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        for p in m.parameters():
            p.requires_grad = True

# Set batchnorm always in eval mode during training
def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def set_bn_train(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.train()