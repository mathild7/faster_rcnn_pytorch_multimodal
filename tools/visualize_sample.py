#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------
"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torchvision.ops import nms

from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse

import torch

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='nuscenes data viewer')
    parser.add_argument(
        '--filename',
        dest='filename',
        help='File that you want to view annotations over',
        default='None')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    # model path
    demonet = args.filename
    saved_model = os.path.join(
        'output', demonet, DATASETS[dataset][0], 'default',
        NETS[demonet][0] % (70000 if dataset == 'pascal_voc' else 110000))
