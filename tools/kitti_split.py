import _init_paths
import tensorflow as tf
import os
import math
import numpy as np
import itertools
import json
from PIL import Image, ImageDraw
from enum import Enum
from model.config import cfg
import utils.bbox as bbox_utils


my_dir = '/home/mat/thesis/data/kitti'
split_dir = os.path.join(my_dir,'splits')
test_split = open(split_dir+'/test.txt').readlines()
train_split = open(split_dir+'/train.txt').readlines()
val_split = open(split_dir+'/val.txt').readlines()