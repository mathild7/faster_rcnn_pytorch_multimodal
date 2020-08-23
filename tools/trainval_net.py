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
from datasets.kitti_lidb import kitti_lidb
from datasets.nuscenes_imdb import nuscenes_imdb
from datasets.waymo_imdb import waymo_imdb
from datasets.waymo_lidb import waymo_lidb
from datasets.cadc_imdb import cadc_imdb
from datasets.cadc_lidb import cadc_lidb
import argparse
import pprint
import numpy as np
import sys
import os
import roi_data_layer.roidb as rdl_roidb
from nets.vgg16 import vgg16
from nets.imagenet import imagenet
from nets.lidarnet import lidarnet
from nets.mobilenet_v1 import mobilenetv1

#https://stackoverflow.com/questions/46561390/4-step-alternating-rpn-faster-r-cnn-training-tensorflow-object-detection-mo/46981671#46981671
#https://arthurdouillard.com/post/faster-rcnn/
#Alternate sharing: 
    #Similar to some matrix decomposition methods, the authors train RPN, then Fast-RCN, and so on. Each network is trained a bit alternatively.
#Approximate joint training: 
    #This strategy consider the two networks as a single unified one. The back-propagation uses both the Fast-RCNN loss and the RPN loss. However the regression of bounding-box coordinates in RPN is considered as pre-computed, and thus its derivative is ignored.
#Non-approximate joint training:
    #This solution was not used as more difficult to implement. The RoI pooling is made differentiable w.r.t the box coordinates using a RoI warping layer.
#4-Step Alternating training: 
    #The strategy chosen takes 4 steps: In the first of one the RPN is trained. In the second, Fast-RCNN is trained using pre-computed RPN proposals. For the third step, the trained Fast-RCNN is used to initialize a new RPN where only RPN’s layers are fine-tuned. Finally in the fourth step RPN’s layers are frozen and only Fast-RCNN is fine-tuned.

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
        '--weights_file',
        dest='weights_file',
        help='initialize with pretrained model weights',
        default=None,
        type=str)
    parser.add_argument(
        '--data_dir',
        dest='data_dir',
        help='root location of all datasets',
        default=None,
        type=str)
    parser.add_argument(
        '--cache_dir',
        dest='cache_dir',
        help='Specify alternate cache directory (not with datasets)',
        default=None,
        type=str)
    parser.add_argument(
        '--db',
        dest='db_name',
        help='dataset to train on',
        default='voc_2007_trainval',
        type=str)
    parser.add_argument(
        '--dbval',
        dest='dbval_name',
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
        '--batch_size',
        dest='train_batch_size',
        help='number of batches to train before backprop',
        default=None,
        type=int)
    parser.add_argument(
        '--batch_size_val',
        dest='trainval_batch_size',
        help='number of batches to validate before backprop',
        default=None,
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
    parser.add_argument(
        '--net_type',
        dest='net_type',
        help='lidar or camera',
        type=str)
    parser.add_argument(
        '--en_full_net',
        dest='en_full_net',
        help='enable just first stage or both stages of net',
        default=1,
        type=int)
    parser.add_argument(
        '--en_fpn',
        dest='en_fpn',
        help='enable FPN',
        default=0,
        type=int)
    parser.add_argument(
        '--en_epistemic',
        dest='en_epistemic',
        help='enable epistemic uncertainty estimation',
        default=0,
        type=int)
    parser.add_argument(
        '--en_aleatoric',
        dest='en_aleatoric',
        help='enable aleatoric uncertainty estimation',
        default=0,
        type=int)
    parser.add_argument(
        '--uc_sort_type',
        dest='uc_sort_type',
        help='Specify the uncertainty type to sort by for drawing evaluation frames',
        default=None,
        type=str)
    parser.add_argument(
        '--iter',
        dest='iter',
        help='what specific folder to save in',
        default=None,
        type=int)
    parser.add_argument(
        '--preload',
        dest='preload',
        help='0: None, 1: preload 1st stage 2: preload full net',
        default=None,
        type=int)
    parser.add_argument(
        '--fixed_blocks',
        dest='fixed_blocks',
        help='(-1,0,1,2,3,4) number of fixed resnet blocks, -1 also enables all batch norm training',
        default=None,
        type=int)
    parser.add_argument(
        '--scale',
        dest='scale',
        help='scale factor for frame',
        default=None,
        type=float)
    if len(sys.argv) == 1 and manual_mode is False:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def get_training_validation_roidb(mode,db,draw_and_save=False):
    """Returns a roidb (Region of Interest database) for use in training."""
    if(mode == 'train'):
        roidb_dummy = db.roidb
    elif(mode == 'val'):
        roidb_dummy = db.val_roidb
    #if cfg.TRAIN.USE_FLIPPED and mode == 'train':
    #    print('Appending horizontally-flipped training examples...')
    #    imdb.append_flipped_images(mode)
    #    print('done')
    
    # Useless
    #print('Preparing ROIs per frame... ')
    #rdl_roidb.prepare_roidb(mode,db)
    #print('done')
    
    if(draw_and_save):
        if(db.type == 'lidar'):
            print('drawing and saving lidar BEV')
        elif(db.type == 'image'):
            print('drawing and saving image')
        db.draw_and_save(mode)
    if(mode == 'train'):
        return db.roidb
    elif(mode == 'val'):
        return db.val_roidb
    else:
        return None

def combined_lidb_roidb(mode,dataset,draw_and_save=False,lidb=None,limiter=0):
    """
  Combine multiple roidbs
  """
    if(mode == 'train'):
        if(dataset == 'waymo'):
            lidb = waymo_lidb(mode,limiter)
        elif(dataset == 'kitti'):
            lidb = kitti_lidb(mode,limiter)
        elif(dataset == 'cadc'):
            lidb = cadc_lidb(mode,limiter)
        else:
            print('Requested dataset is not available')
            return
        print('Loaded dataset `{:s}` for training'.format(lidb.name))
        #Use gt_roidb located in kitti_imdb.py
        #imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
        print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
    print('getting ROIDB Ready for mode {:s}'.format(mode))
    roidb = get_training_validation_roidb(mode,lidb,draw_and_save)

    return lidb, roidb


def combined_imdb_roidb(mode,dataset,draw_and_save=False,imdb=None,limiter=0):
    """
  Combine multiple roidbs
  """
    if(mode == 'train'):
        if(dataset == 'kitti'):
            imdb = kitti_imdb(mode)
        elif(dataset == 'nuscenes'):
            imdb = nuscenes_imdb(mode,limiter)
        elif(dataset == 'waymo'):
            imdb = waymo_imdb(mode,limiter)
        elif(dataset == 'cadc'):
            imdb = cadc_imdb(mode,limiter)
        else:
            print('Requested dataset is not available')
            return None
        print('Loaded dataset `{:s}` for training'.format(imdb.name))
        #Use gt_roidb located in kitti_imdb.py
        #imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
        print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
    print('getting ROIDB Ready for mode {:s}'.format(mode))
    roidb = get_training_validation_roidb(mode,imdb,draw_and_save)

    return imdb, roidb


if __name__ == '__main__':
    cfg.DEBUG.EN = True
    #manual_mode = cfg.DEBUG.EN
    manual_mode = True
    args = parse_args(manual_mode)
    #TODO: Config new image size
    if(manual_mode):
        args.net = 'res101'
        args.db_name = 'kitti'
        #args.out_dir = 'output/'
        args.net_type       = 'lidar'
        args.preload        = 0
        args.iter           = 21
        args.scale          = 1.0
        args.en_full_net    = False
        args.en_fpn         = False
        args.fixed_blocks   = -1
        args.batch_size     = 4
        args.val_batch_size = 32
        #args.en_epistemic = 1
        #args.en_aleatoric = 1
        #args.uc_sort_type = 'a_bbox_var'
        args.fixed_blocks = 0
        args.data_dir     = os.path.join('/home/mat','thesis', 'data')
        #args.uc_sort_type = 'a_bbox_var'
        #args.db_root_dir = '/home/mat/thesis/data/{}/'.format(args.db_name)
        #LIDAR
        #args.weights_file  = os.path.join('/home/mat/thesis/data2/waymo/', 'weights', 'lidar_rpn_75k.pth')
        #args.weights_file  = os.path.join('/home/mat/thesis/data/cadc/', 'weights', 'aug21' ,'lidar_rpn_20k.pth')
        #args.weights_file  = os.path.join('/home/mat/thesis/data2/', args.db_name, 'weights','aug06','image_diag_area_145k.pth')
        #args.weights_file  = os.path.join('/home/mat/thesis/data2/', args.db_name, 'weights', 'image_base_65k.pth')
        #IMAGE
        #args.weights_file  = os.path.join('/home/mat/thesis/data2/', 'stock_weights', 'res101_coco_tf_fpn_1190k.pth')
        #args.weights_file = os.path.join('/home/mat/thesis/data/', 'weights', '{}-caffe.pth'.format(args.net))
        #args.imdbval_name = 'evaluation'
        args.max_iters = 700000
    cfg.DB_NAME = args.db_name
    #TODO: Merge into cfg_from_list()
    if(args.cache_dir is not None):
        cfg.CACHE_DIR = args.cache_dir
    if(args.data_dir is not None):
        cfg.DATA_DIR = args.data_dir
    if(args.fixed_blocks is not None):
        cfg.RESNET.FIXED_BLOCKS = args.fixed_blocks
    if(args.net_type is not None):
        cfg.NET_TYPE = args.net_type
    if(args.en_full_net is not None):
        if(args.en_full_net == 1):
            cfg.ENABLE_FULL_NET = True
        elif(args.en_full_net == 0):
            cfg.ENABLE_FULL_NET = False
    if(args.train_batch_size is not None):
        cfg.TRAIN.BATCH_SIZE = args.train_batch_size
    if(args.trainval_batch_size is not None):
        cfg.TRAIN.BATCH_SIZE_VAL = args.trainval_batch_size
    if(args.iter is not None):
        cfg.TRAIN.ITER = args.iter
    if(args.preload is not None):
        cfg.PRELOAD     = False
        cfg.PRELOAD_FULL = False
        if(args.preload == 1):
            cfg.PRELOAD     = True
        elif(args.preload == 2):
            cfg.PRELOAD_FULL = True
    if(args.en_fpn == 1):
        cfg.USE_FPN = True
        cfg.TRAIN.ROI_BATCH_SIZE = 256
        cfg.POOLING_MODE = 'multiscale'
        cfg.ENABLE_CUSTOM_TAIL = True
    else:
        cfg.TRAIN.ROI_BATCH_SIZE = 256
        cfg.POOLING_MODE = 'align'
    if(args.weights_file is None and cfg.ENABLE_FULL_NET):
        if(cfg.NET_TYPE == 'lidar'):
            args.weights_file  = os.path.join('/home/mat/thesis/data/', 'weights', 'res101_lidar_full_100p_136k.pth')
        elif(cfg.NET_TYPE == 'image'):
            #args.weights_file = os.path.join('/home/mat/thesis/data2/', 'stock_weights', '{}_coco_tf_fpn_1190k.pth'.format(args.net))
            args.weights_file = os.path.join('/home/mat/thesis/data/', 'weights', '{}-caffe.pth'.format(args.net))
    if(args.scale is not None):
        cfg.TRAIN.SCALES = (args.scale,)

    if(args.en_epistemic == 1):
        cfg.UC.EN_BBOX_EPISTEMIC = True
        cfg.UC.EN_CLS_EPISTEMIC  = True
    if(args.en_aleatoric == 1):
        cfg.UC.EN_BBOX_ALEATORIC = True
        cfg.UC.EN_CLS_ALEATORIC  = True
    if(args.uc_sort_type is not None):
        cfg.UC.SORT_TYPE = args.uc_sort_type

    print('Called with args:')
    print(args)
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    np.random.seed(cfg.RNG_SEED)
    # train set
    if(cfg.NET_TYPE == 'image'):
        db, roidb     = combined_imdb_roidb('train',args.db_name,cfg.TRAIN.DRAW_ROIDB_GEN,None,limiter=0)
        _, val_roidb  = combined_imdb_roidb('val',args.db_name,cfg.TRAIN.DRAW_ROIDB_GEN,db,limiter=0)
    elif(cfg.NET_TYPE == 'lidar'):
        db, roidb     = combined_lidb_roidb('train',args.db_name,cfg.TRAIN.DRAW_ROIDB_GEN,None,limiter=0)
        _, val_roidb  = combined_lidb_roidb('val',args.db_name,cfg.TRAIN.DRAW_ROIDB_GEN,db,limiter=0)
    print('{:d} roidb entries'.format(len(roidb)))
    print('{:d} val roidb entries'.format(len(val_roidb)))
    # output directory where the models are saved
    output_dir = get_output_dir(db, weights_filename=args.tag)
    print('Output will be saved to `{:s}`'.format(output_dir))

    # tensorboard directory where the summaries are saved during training
    tb_dir = get_output_tb_dir(db, weights_filename=args.tag)
    print('TensorFlow summaries will be saved to `{:s}`'.format(tb_dir))

    # also add the validation set, but with no flipping images
    #orgflip = cfg.TRAIN.USE_FLIPPED
    #cfg.TRAIN.USE_FLIPPED = False
    print('db: {}'.format(args.db_name))
    #cfg.TRAIN.USE_FLIPPED = orgflipqua

    # load network
    if(cfg.NET_TYPE == 'image'):
        if args.net == 'vgg16':
            net = vgg16()
        elif args.net == 'res34':
            net = imagenet(num_layers=34)
        elif args.net == 'res50':
            net = imagenet(num_layers=50)
        elif args.net == 'res101':
            net = imagenet(num_layers=101)
        elif args.net == 'res152':
            net = imagenet(num_layers=152)
        elif args.net == 'mobile':
            net = mobilenetv1()
        else:
            raise NotImplementedError
    elif(cfg.NET_TYPE == 'lidar'):
        net = lidarnet(num_layers=101)
        
    train_net(
        net,
        db,
        output_dir,
        tb_dir,
        pretrained_model=args.weights_file,
        max_iters=args.max_iters,
        sum_size=256,
        val_sum_size=5000,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        val_batch_size=cfg.TRAIN.VAL_BATCH_SIZE,
        val_thresh=0.4,
        augment_en=cfg.TRAIN.AUGMENT_EN,
        val_augment_en=cfg.TRAIN.VAL_AUGMENT_EN)
