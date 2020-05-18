from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C



#Need to turn this on in order to debug
__C.DEBUG                    = edict()

__C.DEBUG.EN                 = False

__C.DEBUG.DRAW_ANCHORS       = False
__C.DEBUG.DRAW_ANCHOR_T      = False
__C.DEBUG.DRAW_PROPOSAL_T    = False
__C.DEBUG.DRAW_MINIBATCH     = False
__C.DEBUG.TEST_FRAME_PRINT   = False
__C.DEBUG.FREEZE_DB          = False
__C.DEBUG.FREEZE_DB_INDS     = 1
__C.DEBUG.PRINT_SCENE_RESULT = False

#Bayesian Config
__C.UC = edict()
__C.UC.EN_RPN_BBOX_ALEATORIC = False
__C.UC.EN_RPN_CLS_ALEATORIC  = False
__C.UC.EN_RPN_BBOX_EPISTEMIC = False
__C.UC.EN_RPN_CLS_EPISTEMIC  = False
__C.UC.EN_BBOX_ALEATORIC     = False
__C.UC.EN_CLS_ALEATORIC      = False
__C.UC.EN_BBOX_EPISTEMIC     = False
__C.UC.EN_CLS_EPISTEMIC      = False
__C.UC.A_NUM_CE_SAMPLE       = 80
__C.UC.A_NUM_BBOX_SAMPLE     = 80
__C.UC.E_NUM_SAMPLE          = 80
__C.UC.SORT_TYPE             = ''
#ONE OF
__C.PRELOAD                  = False
__C.PRELOAD_FULL             = False
__C.USE_FPN = False
__C.ENABLE_FULL_NET          = True
__C.NET_TYPE                 = 'lidar'
__C.SCALE_LOC                = 6
#WAYMO input size
__C.IM_SIZE = [1920,930]
#
# Training options
#
__C.TRAIN = edict()

# Initial learning rate
#WAYMO
__C.TRAIN.LEARNING_RATE = 0.01
#Kitti
#__C.TRAIN.LEARNING_RATE = 0.001
# Momentum
__C.TRAIN.MOMENTUM = 0.6

# Weight decay, for regularization
#WAYMO
__C.TRAIN.WEIGHT_DECAY = 0.0003
#__C.TRAIN.WEIGHT_DECAY = 0.0001
# Factor for reducing the learning rate
__C.TRAIN.GAMMA = 0.1

# Step size for reducing the learning rate, currently only support one step
#KITTI ~7,000 images in train set
__C.TRAIN.STEPSIZE = [70000, 140000, 210000]
#NUSCENES ~50,000 images in train set
#__C.TRAIN.STEPSIZE = [300000, 500000, 700000]
#WAYMO ~15,000 images in train set
#__C.TRAIN.STEPSIZE = [20000,40000,60000,70000,80000]
# Iteration intervals for showing the loss during training, on command line interface
__C.TRAIN.DISPLAY = 512

# Whether to double the learning rate for bias
__C.TRAIN.DOUBLE_BIAS = False

# Whether to initialize the weights with truncated normal distribution 
__C.TRAIN.TRUNCATED = False

# Whether to have weight decay on bias as well
__C.TRAIN.BIAS_DECAY = False

# Whether to add ground truth boxes to the pool when sampling regions
__C.TRAIN.USE_GT = False

# The number of snapshots kept, older ones are deleted to save space
__C.TRAIN.SNAPSHOT_KEPT = 30

# The time interval for saving tensorflow summaries
__C.TRAIN.SUMMARY_INTERVAL = 15

# Scale to use during training (can list multiple scales)
__C.TRAIN.SCALES = (1.0,)

# Images/Lidar frames to use per minibatch
__C.TRAIN.FRAMES_PER_BATCH = 1

# Minibatch size (number of regions of interest [ROIs])
__C.TRAIN.BATCH_SIZE = 128

# Fraction of minibatch that is labeled foreground (i.e. class > 0)
__C.TRAIN.FG_FRACTION = 0.25

# Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
__C.TRAIN.FG_THRESH = 0.5
__C.TRAIN.DC_THRESH = 0.5
# Overlap threshold for a ROI to be considered background (class = 0 if
# overlap in [LO, HI))
__C.TRAIN.BG_THRESH_HI = 0.5
__C.TRAIN.BG_THRESH_LO = 0.1

# Use horizontally-flipped images during training?
__C.TRAIN.USE_FLIPPED = True

# Train bounding-box regressors
__C.TRAIN.BBOX_REG = True

# Overlap required between a ROI and ground-truth box in order for that ROI to
# be used as a bounding-box regression training example
__C.TRAIN.BBOX_THRESH = 0.5

# Iterations between snapshots
__C.TRAIN.SNAPSHOT_ITERS = 5000

# solver.prototxt specifies the snapshot path prefix, this adds an optional
# infix to yield the path: <prefix>[_<infix>]_iters_XYZ.caffemodel
__C.TRAIN.SNAPSHOT_PREFIX = 'res101_faster_rcnn'

# Normalize the targets (subtract empirical mean, divide by empirical stddev)
__C.TRAIN.BBOX_NORMALIZE_TARGETS = True

# Deprecated (inside weights)
__C.TRAIN.BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)

# Normalize the targets using "precomputed" (or made up) means and stdevs
# (BBOX_NORMALIZE_TARGETS must also be True)
__C.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED  = True

# Train using these proposals
__C.TRAIN.PROPOSAL_METHOD = 'gt'

# Make minibatches from images that have similar aspect ratios (i.e. both
# tall and thin or both short and wide) in order to avoid wasting computation
# on zero-padding.

# Use RPN to detect objects
__C.TRAIN.HAS_RPN = True

# IOU >= thresh: positive example
__C.TRAIN.RPN_POSITIVE_OVERLAP = 0.7

# IOU < thresh: negative example
__C.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3

# If an anchor satisfied by positive and negative conditions set to negative
__C.TRAIN.RPN_CLOBBER_POSITIVES = False

# Max number of foreground examples
__C.TRAIN.RPN_FG_FRACTION = 0.5

# Total number of examples
__C.TRAIN.RPN_BATCHSIZE = 256

# NMS threshold used on RPN proposals
__C.TRAIN.RPN_NMS_THRESH = 0.7

# Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TRAIN.RPN_PRE_NMS_TOP_N = 12000

# Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TRAIN.RPN_POST_NMS_TOP_N = 2000

# Deprecated (outside weights)
__C.TRAIN.RPN_BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)

# Give the positive RPN examples weight of p * 1 / {num positives}
# and give negatives a weight of (1 - p)
# Set to -1.0 to use uniform example weighting
__C.TRAIN.RPN_POSITIVE_WEIGHT = -1.0

# Whether to use all ground truth bounding boxes for training, 
# For COCO, setting USE_ALL_GT to False will exclude boxes that are flagged as ''iscrowd''
__C.TRAIN.USE_ALL_GT = True

#Whether or not to ignore dont care areas when training
__C.TRAIN.IGNORE_DC = False

__C.TRAIN.ITER = 1

__C.TRAIN.LIDAR = edict()

__C.TRAIN.IMAGE = edict()
__C.TRAIN.LIDAR.BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
__C.TRAIN.LIDAR.BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2)

__C.TRAIN.IMAGE.BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)
__C.TRAIN.IMAGE.BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)
#
# Testing options
#
__C.TEST = edict()

# Scale to use during testing (can NOT list multiple scales)
__C.TEST.SCALES  = (1.0,)
# Max pixel size of the longest side of a scaled input image
# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
__C.TEST.NMS = 0.4

# Experimental: treat the (K+1) units in the cls_score layer as linear
# predictors (trained, eg, with one-vs-rest SVMs).
__C.TEST.SVM = False

# Test using bounding-box regressors
__C.TEST.BBOX_REG = True

# Propose boxes
__C.TEST.HAS_RPN = True

# Test using these proposals
__C.TEST.PROPOSAL_METHOD = 'gt'

## NMS threshold used on RPN proposals
__C.TEST.RPN_NMS_THRESH = 0.7

# Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TEST.RPN_PRE_NMS_TOP_N = 6000

# Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TEST.RPN_POST_NMS_TOP_N = 300

# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
# __C.TEST.RPN_MIN_SIZE = 16

# Testing mode, default to be 'nms', 'top' is slower but better
# See report for details
__C.TEST.MODE = 'nms'

# Only useful when TEST.MODE is 'top', specifies the number of top proposals to select
__C.TEST.RPN_TOP_N = 5000

__C.TEST.IGNORE_DC = False

__C.TEST.ITER = 1
#
# ResNet options
#

__C.RESNET = edict()

# Option to set if max-pooling is appended after crop_and_resize. 
# if true, the region will be resized to a square of 2xPOOLING_SIZE, 
# then 2x2 max-pooling is applied; otherwise the region will be directly
# resized to a square of POOLING_SIZE
__C.RESNET.MAX_POOL = False

# Number of fixed blocks during training, by default the first of all 4 blocks is fixed
# Range: -1 (none) to 3 (all)
__C.RESNET.FIXED_BLOCKS = 1

#
# MobileNet options
#

__C.MOBILENET = edict()

# Whether to regularize the depth-wise filters during training
__C.MOBILENET.REGU_DEPTH = False

# Number of fixed layers during training, by default the bottom 5 of 14 layers is fixed
# Range: 0 (none) to 12 (all)
__C.MOBILENET.FIXED_LAYERS = 5

# Weight decay for the mobilenet weights
__C.MOBILENET.WEIGHT_DECAY = 0.00004

# Depth multiplier
__C.MOBILENET.DEPTH_MULTIPLIER = 1.0

#
# MISC
#

# Pixel mean values (BGR order) as a (1, 1, 3) array
# We use the same pixel mean for all networks even though it's not exactly what
# they were trained with
#__C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
#UPDATED for Kitti 
#__C.PIXEL_MEANS = np.array([[[96.866, 98.76, 93.85]]])
#__C.PIXEL_STDDEVS = np.array([[[81.54, 80.50, 79.466]]])
#__C.PIXEL_ARRANGE = [0,1,2]
#UPDATED for waymo
# For reproducibility
#__C.PIXEL_MEANS = np.array([[[72.03, 75.20, 80.18]]])
#__C.PIXEL_STDDEVS = np.array([[[26, 28, 38]]])
#__C.PIXEL_STDDEVS = np.array([[[60, 60, 60]]])
#__C.PIXEL_STDDEVS = np.array([[[255, 255, 255]]])
#__C.PIXEL_ARRANGE = [0,1,2]

#resenet101-pytorch
#__C.PIXEL_MEANS = np.array([[[123.675, 116.28, 103.53]]])
#__C.PIXEL_MEANS = np.array([[[78.675, 75.28, 70.53]]])
#__C.PIXEL_STDDEVS = np.array([[[58.395,57.12,57.375]]])
#[B,G,R] to [R,G,B]
#__C.PIXEL_ARRANGE = [2,1,0]

#Maximum value to clip at when performing backprop
__C.GRAD_MAX_CLIP = 20

#cafferesnet101
__C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
__C.PIXEL_STDDEVS = np.array([[[1, 1, 1]]])
__C.PIXEL_ARRANGE = [0,1,2]
__C.PIXEL_ARRANGE_BGR = [2,1,0]

__C.RNG_SEED = 3

# Root directory of project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))

# Data directory
__C.DATA_DIR = osp.abspath(osp.join('/home/mat','thesis', 'data2'))

# Name (or path to) the matlab executable
__C.MATLAB = 'matlab'

# Place outputs under an experiments directory
__C.EXP_DIR = 'res101'

# Use GPU implementation of non-maximum suppression
__C.USE_GPU_NMS = True

# Default pooling mode
__C.POOLING_MODE = 'align'

# Size of the pooled region after RoI pooling
__C.POOLING_SIZE = 7

# Anchor scales for RPN
__C.ANCHOR_SCALES = [2,8,16] # 32x32, 64x64, 256x256

# Anchor ratios for RPN
__C.ANCHOR_RATIOS = [0.5,1,2]

# Number of filters for the RPN layer
__C.RPN_CHANNELS = 512


__C.ENABLE_CUSTOM_TAIL       = False
__C.NUM_SCENES               = 210
__C.MAX_IMG_PER_SCENE        = 1000
__C.TRAIN.TOD_FILTER_LIST    = ['Day','Night','Dawn/Dusk']
__C.TRAIN.DRAW_ROIDB_GEN     = False
__C.TEST.TOD_FILTER_LIST     = ['Day','Night','Dawn/Dusk']
#Lidar Config
__C.LIDAR = edict()
__C.LIDAR.X_RANGE            = [0,70]
__C.LIDAR.Y_RANGE            = [-40,40]
__C.LIDAR.Z_RANGE            = [-3,3]
__C.LIDAR.VOXEL_LEN          = 0.1
__C.LIDAR.VOXEL_HEIGHT       = 0.5
__C.LIDAR.NUM_SLICES         = 12
__C.LIDAR.NUM_CHANNEL        = __C.LIDAR.NUM_SLICES + 3
__C.LIDAR.MAX_PTS_PER_VOXEL  = 32
__C.LIDAR.MAX_NUM_VOXEL      = 25000

#height -> R, Intensity -> G, Elongation/Density -> B
#TODO: Broken, dont use..
#__C.LIDAR.MEANS         = np.array([[[102.9801, 102.9801, 102.9801, 102.9801, 102.9801, 102.9801, 102.9801, 102.9801, 115.9465, 122.7717]]])
#__C.LIDAR.STDDEVS       = np.array([[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]])
#(l,w,h) corresponding to (x,y,z)
__C.LIDAR.ANCHORS       = np.array([[4.73,2.08,1.77]])
__C.LIDAR.ANCHOR_SCALES = np.array([[1]])
__C.LIDAR.ANCHOR_ANGLES = np.array([0,np.pi/2])
__C.LIDAR.ANCHOR_STRIDE = np.array([2,2,0.5])
__C.LIDAR.NUM_BBOX_ELEM = 7

__C.IMAGE = edict()
__C.IMAGE.NUM_BBOX_ELEM = 4


def get_output_dir(db, mode='train', weights_filename=None):
  """Return the directory where experimental artifacts are placed.
  If the directory does not exist, it is created.

  A canonical path is built using the name from an imdb and a network
  (if not None).
  """
  outdir = osp.abspath(osp.join(__C.ROOT_DIR, 'output', __C.EXP_DIR, db.name))
  if weights_filename is None:
    net_type = '{}_'.format(__C.NET_TYPE)
    if __C.ENABLE_FULL_NET is False:
      net_type = net_type + 'rpn_only_'
    if __C.UC.EN_BBOX_ALEATORIC:
      net_type = net_type + 'a_bbox_'
    if __C.UC.EN_CLS_ALEATORIC:
      net_type = net_type + 'a_cls_'
    if __C.UC.EN_BBOX_EPISTEMIC:
      net_type = net_type + 'e_bbox_'
    if __C.UC.EN_CLS_EPISTEMIC:
      net_type = net_type + 'e_cls_'
    if __C.UC.EN_RPN_BBOX_ALEATORIC:
      net_type = net_type + 'a_rpn_bbox_'
    if __C.UC.EN_RPN_CLS_ALEATORIC:
      net_type = net_type + 'a_rpn_cls_'
    if __C.UC.EN_RPN_BBOX_EPISTEMIC:
      net_type = net_type + 'e_rpn_bbox_'
    if __C.UC.EN_RPN_CLS_EPISTEMIC:
      net_type = net_type + 'e_rpn_cls_'
    #catch all, if nothing is enabled
    if(net_type == ''):
      net_type = 'vanilla_'
    if(len(__C.TRAIN.TOD_FILTER_LIST) == 3):
      train_filter = 'all'
    elif(__C.TRAIN.TOD_FILTER_LIST[0] == 'Day'):
      train_filter = 'day'
    elif(__C.TRAIN.TOD_FILTER_LIST[0] == 'Night'):
      train_filter = 'night'
    elif(__C.TRAIN.TOD_FILTER_LIST[0] == 'Dawn/Dusk'):
      train_filter = 'dawn_dusk'
    weights_filename = '{}{}_{}_{}'.format(net_type,mode,train_filter,__C[mode.upper()].ITER)
  outdir = osp.join(outdir, weights_filename)
  if not os.path.exists(outdir):
    os.makedirs(outdir)
  return outdir


def get_output_tb_dir(db, weights_filename):
  """Return the directory where tensorflow summaries are placed.
  If the directory does not exist, it is created.

  A canonical path is built using the name from an imdb and a network
  (if not None).
  """
  outdir = osp.abspath(osp.join(__C.ROOT_DIR, 'tensorboard', __C.EXP_DIR, db.name))
  if weights_filename is None:
    net_type = '{}_'.format(__C.NET_TYPE)
    if __C.ENABLE_FULL_NET is False:
      net_type = net_type + 'rpn_only_'
    if __C.UC.EN_BBOX_ALEATORIC:
      net_type = net_type + 'a_bbox_'
    if __C.UC.EN_CLS_ALEATORIC:
      net_type = net_type + 'a_cls_'
    if __C.UC.EN_BBOX_EPISTEMIC:
      net_type = net_type + 'e_bbox_'
    if __C.UC.EN_CLS_EPISTEMIC:
      net_type = net_type + 'e_cls_'
    if __C.UC.EN_RPN_BBOX_ALEATORIC:
      net_type = net_type + 'a_rpn_bbox_'
    if __C.UC.EN_RPN_CLS_ALEATORIC:
      net_type = net_type + 'a_rpn_cls_'
    if __C.UC.EN_RPN_BBOX_EPISTEMIC:
      net_type = net_type + 'e_rpn_bbox_'
    if __C.UC.EN_RPN_CLS_EPISTEMIC:
      net_type = net_type + 'e_rpn_cls_'
    #catch all, if nothing is enabled
    if(net_type == ''):
      net_type = 'vanilla_'
    
    if(len(__C.TRAIN.TOD_FILTER_LIST) == 3):
      train_filter = 'all'
    elif(__C.TRAIN.TOD_FILTER_LIST[0] == 'Day'):
      train_filter = 'day'
    elif(__C.TRAIN.TOD_FILTER_LIST[0] == 'Night'):
      train_filter = 'night'
    elif(__C.TRAIN.TOD_FILTER_LIST[0] == 'Dawn/Dusk'):
      train_filter = 'dawn_dusk'
    weights_filename = '{}train_{}_{}'.format(net_type,train_filter,__C.TRAIN.ITER)
  outdir = osp.join(outdir, weights_filename)
  if not os.path.exists(outdir):
    os.makedirs(outdir)
  return outdir


def _merge_a_into_b(a, b):
  """Merge config dictionary a into config dictionary b, clobbering the
  options in b whenever they are also specified in a.
  """
  if type(a) is not edict:
    return

  for k, v in a.items():
    # a must specify keys that are in b
    if k not in b:
      raise KeyError('{} is not a valid config key'.format(k))

    # the types must match, too
    old_type = type(b[k])
    if old_type is not type(v):
      if isinstance(b[k], np.ndarray):
        v = np.array(v, dtype=b[k].dtype)
      else:
        raise ValueError(('Type mismatch ({} vs. {}) '
                          'for config key: {}').format(type(b[k]),
                                                       type(v), k))

    # recursively merge dicts
    if type(v) is edict:
      try:
        _merge_a_into_b(a[k], b[k])
      except:
        print(('Error under config key: {}'.format(k)))
        raise
    else:
      b[k] = v


def cfg_from_file(filename):
  """Load a config file and merge it into the default options."""
  import yaml
  with open(filename, 'r') as f:
    yaml_cfg = edict(yaml.load(f))

  _merge_a_into_b(yaml_cfg, __C)


def cfg_from_list(cfg_list):
  """Set config keys via list (e.g., from command line)."""
  from ast import literal_eval
  assert len(cfg_list) % 2 == 0
  for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
    key_list = k.split('.')
    d = __C
    for subkey in key_list[:-1]:
      assert subkey in d
      d = d[subkey]
    subkey = key_list[-1]
    assert subkey in d
    try:
      value = literal_eval(v)
    except:
      # handle the case when v is a string literal
      value = v
    assert type(value) == type(d[subkey]), \
      'type {} does not match original type {}'.format(
        type(value), type(d[subkey]))
    d[subkey] = value
