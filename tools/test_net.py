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
from datasets.waymo_lidb import waymo_lidb
from nets.lidarnet import lidarnet
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
        '--db',
        dest='db_name',
        help='dataset to test',
        default='voc_2007_test',
        type=str)
    parser.add_argument(
        '--db_root_dir',
        dest='db_root_dir',
        help='location of dataset',
        default='/',
        type=str)
    parser.add_argument(
        '--db_out_dir',
        dest='out_dir',
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
        dest='max_num_dets',
        help='max number of detections per frame',
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
    parser.add_argument(
        '--net_type',
        dest='net_type',
        help='lidar or camera',
        type=str)
    parser.add_argument(
        '--iter',
        dest='iter',
        help='what specific folder to save in',
        default=None,
        type=int)
    parser.add_argument(
        '--num_frames',
        dest='num_frames',
        help='how many frames to test',
        default=0,
        type=int)
    parser.add_argument(
        '--scale',
        dest='scale',
        help='scale factor for frame input ',
        default=None,
        type=float)

    if len(sys.argv) == 1 and manual_mode is False:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    print(args)
    return args


if __name__ == '__main__':
    manual_mode = False
    args = parse_args(manual_mode)
    if(manual_mode):
        args.net = 'res101'
        args.db_name = 'waymo'
        args.net_type = 'lidar'
        args.weights_file = '{}_{}_100p_180k.pth'.format(args.net,args.net_type)
        args.iter = 10
        args.num_frames = 0
        args.scale = 1.0
        #args.out_dir = 'output/'
        #args.db_root_dir = '/home/mat/thesis/data2/{}/'.format(args.db_name)
    print('Called with args:')
    print(args)

    #TODO: Merge into cfg_from_list()
    if(args.net_type is not None):
        cfg.NET_TYPE = args.net_type
    if(args.iter is not None):
        cfg.TEST.ITER = args.iter
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    if(args.out_dir is None):
        if(cfg.NET_TYPE == 'lidar'):
            args.out_dir = 'lidar_test_{}'.format(cfg.TEST.ITER)
        elif(cfg.NET_TYPE == 'image'):
            args.out_dir = 'image_test_{}'.format(cfg.TEST.ITER)
    if(args.scale is not None):
        cfg.TEST.SCALES = (args.scale,)

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
    if(cfg.NET_TYPE == 'image'):
        if(args.db_name == 'kitti'):
            db = kitti_imdb(mode='eval')
        elif(args.db_name == 'nuscenes'):
            db = nuscenes_imdb(mode='val',limiter=args.num_frames)
        elif(args.db_name == 'waymo'):
            db = waymo_imdb(mode='val',limiter=args.num_frames, shuffle_en=True)
    elif(cfg.NET_TYPE == 'lidar'):
        db = waymo_lidb(mode='val',limiter=args.num_frames, shuffle_en=True)

    # load network
    if(cfg.NET_TYPE == 'image'):
        if args.net == 'vgg16':
            net = vgg16()
        elif args.net == 'res34':
            net = resnetv1(num_layers=34)
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
    elif(cfg.NET_TYPE == 'lidar'):
        net = lidarnet(num_layers=101)
    # load model
    if(cfg.NET_TYPE == 'lidar'):
        #TODO: Magic numbers, need to sort this out to flow through 3d anchor gen properly
        net.create_architecture(
            db.num_classes,
            tag='default',
            anchor_scales=cfg.LIDAR.ANCHOR_SCALES,
            anchor_ratios=cfg.LIDAR.ANCHOR_ANGLES)
    elif(cfg.NET_TYPE == 'image'):
        net.create_architecture(
            db.num_classes,
            tag='default',
            anchor_scales=cfg.ANCHOR_SCALES,
            anchor_ratios=cfg.ANCHOR_RATIOS)

    net.eval()

    print(('Loading initial weights from {:s}').format(args.weights_file))
    file_dir = os.path.join(cfg.DATA_DIR,args.db_name,'weights',args.weights_file)
    params = torch.load(file_dir, map_location=lambda storage, loc: storage)
    net.load_state_dict(params)
    print('Loaded.')
    if not torch.cuda.is_available():
        net._device = 'cpu'
    net.to(net._device)
    #TODO: Fix stupid output directory bullshit
    test_net(net, db, args.out_dir, max_dets=args.max_num_dets, mode='val',thresh=0.70,draw_det=False,eval_det=True)
