# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi he, Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Unused
# import time
import _init_paths
from model.test import test_net
from model.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
import argparse
import pprint
import os
import sys
from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
from nets.mobilenet_v1 import mobilenetv1
from datasets.kitti_imdb import kitti_imdb
from datasets.nuscenes_imdb import nuscenes_imdb
from datasets.waymo_imdb import waymo_imdb
import torch

def parse_args(manual_mode):
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='optional config file',
        default=None,
        type=str)
    parser.add_argument(
        '--model', dest='model', help='model to test', default=None, type=str)
    parser.add_argument(
        '--imdb',
        dest='imdb_name',
        help='dataset to test',
        default='voc_2007_test',
        type=str)
    parser.add_argument(
        '--imdb_root_dir',
        dest='imdb_root_dir',
        help='location of dataset',
        default='/',
        type=str)
    parser.add_argument(
        '--imdb_out_dir',
        dest='our_dir',
        help='location of output dir',
        default='/',
        type=str)
    parser.add_argument(
        '--comp',
        dest='comp_mode',
        help='competition mode',
        action='store_true')
    parser.add_argument(
        '--weights_file',
        dest='weights_file',
        help='location of weights file to be loaded',
        type=str)
    parser.add_argument(
        '--num_dets',
        dest='max_per_image',
        help='max number of detections per image',
        default=100,
        type=int)
    parser.add_argument(
        '--tag', dest='tag', help='tag of the model', default='', type=str)
    parser.add_argument(
        '--net',
        dest='net',
        help='vgg16, res50, res101, res152, mobile',
        default='res50',
        type=str)
    parser.add_argument(
        '--set',
        dest='set_cfgs',
        help='set config keys',
        default=None,
        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1 and manual_mode is False:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    print(args)
    return args


if __name__ == '__main__':
    manual_mode = True
    args = parse_args(manual_mode)
    if(manual_mode):
        args.net = 'res101'
        args.imdb_name = 'waymo'
        args.weights_file = 'weights/{}_faster_rcnn_iter_130000.pth'.format(args.net)
        args.out_dir = 'output/'
        args.imdb_root_dir = '/home/mat/thesis/data/{}/'.format(args.imdb_name)
    print('Called with args:')
    print(args)
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)
    # if has model, get the name from it
    # if does not, then just use the initialization weights
    #if args.model:
    #    filename = os.path.splitext(os.path.basename(args.model))[0]
    #else:
    #    filename = os.path.splitext(os.path.basename(args.weights_file))[0]

    #tag = args.tag
    #tag = tag if tag else 'default'
    #filename = tag + '/' + filename
    if(args.imdb_name == 'kitti'):
        imdb = kitti_imdb(mode='eval')
    elif(args.imdb_name == 'nuscenes'):
        imdb = nuscenes_imdb(mode='val',limiter=1000)
    elif(args.imdb_name == 'waymo'):
        imdb = waymo_imdb(mode='val',limiter=1000, shuffle_en=True,tod_filter_list=cfg.TEST.TOD_FILTER_LIST)

    # load network
    if args.net == 'vgg16':
        net = vgg16()
    elif args.net == 'res50':
        net = resnetv1(num_layers=50)
    elif args.net == 'res101':
        net = resnetv1(num_layers=101)
    elif args.net == 'res152':
        net = resnetv1(num_layers=152)
    elif args.net == 'mobile':
        net = mobilenetv1()
    else:
        raise NotImplementedError

    # load model
    net.create_architecture(
        imdb.num_classes,
        tag='default',
        anchor_scales=cfg.ANCHOR_SCALES,
        anchor_ratios=cfg.ANCHOR_RATIOS)

    net.eval()

    print(('Loading initial weights from {:s}').format(args.weights_file))
    params = torch.load(args.imdb_root_dir + args.weights_file, map_location=lambda storage, loc: storage)
    net.load_state_dict(params)
    print('Loaded.')
    if not torch.cuda.is_available():
        net._device = 'cpu'
    net.to(net._device)
    #TODO: Fix stupid output directory bullshit
    test_net(net, imdb, args.out_dir, max_per_image=args.max_per_image, mode='val',thresh=0.6,draw_det=True,eval_det=True)
