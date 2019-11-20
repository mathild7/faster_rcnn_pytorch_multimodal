# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi He, Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.train_val import train_net
from model.config import cfg, cfg_from_file, cfg_from_list, get_output_dir, get_output_tb_dir
from datasets.factory import get_imdb
from datasets.kitti_imdb import kitti_imdb
from datasets.nuscenes_imdb import nuscenes_imdb
import datasets.imdb
import argparse
import pprint
import numpy as np
import sys
import os
import roi_data_layer.roidb as rdl_roidb

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
from nets.mobilenet_v1 import mobilenetv1


def parse_args(manual_mode=False):
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='optional config file',
        default=None,
        type=str)
    parser.add_argument(
        '--weight',
        dest='weight',
        help='initialize with pretrained model weights',
        type=str)
    parser.add_argument(
        '--imdb',
        dest='imdb_name',
        help='dataset to train on',
        default='voc_2007_trainval',
        type=str)
    parser.add_argument(
        '--imdbval',
        dest='imdbval_name',
        help='dataset to validate on',
        default='voc_2007_test',
        type=str)
    parser.add_argument(
        '--iters',
        dest='max_iters',
        help='number of iterations to train',
        default=70000,
        type=int)
    parser.add_argument(
        '--tag', dest='tag', help='tag of the model', default=None, type=str)
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
    return args


def get_training_validation_roidb(mode,imdb,draw_and_save=False):
    """Returns a roidb (Region of Interest database) for use in training."""
    if(mode == 'train'):
        roidb_dummy = imdb.roidb
    elif(mode == 'val'):
        roidb_dummy = imdb.val_roidb
    if cfg.TRAIN.USE_FLIPPED and mode == 'train':
        print('Appending horizontally-flipped training examples...')
        imdb.append_flipped_images(mode)
        print('done')
    print('Preparing ROIs per image... ')
    rdl_roidb.prepare_roidb(mode,imdb)
    print('done')
    if(draw_and_save):
        print('drawing and saving images')
        imdb.draw_and_save(mode)
    if(mode == 'train'):
        return imdb.roidb
    elif(mode == 'val'):
        return imdb.val_roidb
    else:
        return None

def combined_roidb(mode,dataset,draw_and_save=False,imdb=None,limiter=0):
    """
  Combine multiple roidbs
  """
    if(mode == 'train'):
        if(dataset == 'kitti'):
            imdb = kitti_imdb(mode)
        elif(dataset == 'nuscenes'):
            imdb = nuscenes_imdb(mode,limiter)
        else:
            print('Requested dataset is not available')
            return
        print('Loaded dataset `{:s}` for training'.format(imdb.name))
        #Use gt_roidb located in kitti_imdb.py
        #imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
        print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
    print('getting ROIDB Ready for mode {:s}'.format(mode))
    roidb = get_training_validation_roidb(mode,imdb,draw_and_save)

    return imdb, roidb


if __name__ == '__main__':
    manual_mode = True
    args = parse_args(manual_mode)
    #TODO: Config new image size
    if(manual_mode):
        args.net = 'res50'
        args.imdb_name = 'nuscenes'
        args.out_dir = 'output/'
        args.imdb_root_dir = '/home/mat/thesis/data/nuscenes/'
        args.weight = os.path.join('/home/mat/thesis/data/', 'weights', 'resnet50-caffe.pth')
        #args.imdbval_name = 'evaluation'
        args.max_iters = 1000000
    print('Called with args:')
    print(args)
    draw_and_save = True
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    np.random.seed(cfg.RNG_SEED)

    # train set
    imdb, roidb = combined_roidb('train',args.imdb_name,draw_and_save,None,limiter=0)
    _ , val_roidb = combined_roidb('val',args.imdb_name,draw_and_save,imdb,limiter=0)
    #TODO: Shuffle images

    #print(roidb[0])
    print('{:d} roidb entries'.format(len(roidb)))
    print('{:d} val roidb entries'.format(len(val_roidb)))
    # output directory where the models are saved
    output_dir = get_output_dir(imdb, args.tag)
    print('Output will be saved to `{:s}`'.format(output_dir))

    # tensorboard directory where the summaries are saved during training
    tb_dir = get_output_tb_dir(imdb, args.tag)
    print('TensorFlow summaries will be saved to `{:s}`'.format(tb_dir))

    # also add the validation set, but with no flipping images
    orgflip = cfg.TRAIN.USE_FLIPPED
    cfg.TRAIN.USE_FLIPPED = False
    print('imdb')
    print(args.imdb_name)
    cfg.TRAIN.USE_FLIPPED = orgflip

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
    train_net(
        net,
        imdb,
        output_dir,
        tb_dir,
        pretrained_model=args.weight,
        max_iters=args.max_iters,
        sum_size=128,
        val_sum_size=1000,
        batch_size=16,
        val_im_thresh=0.3)