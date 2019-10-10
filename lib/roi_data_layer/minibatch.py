# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
"""Compute minibatch blobs for training a Fast R-CNN network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import numpy as np
import numpy.random as npr
import cv2
from model.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob
import matplotlib.pyplot as plt


def get_minibatch(roidb, num_classes):
    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(
        0, high=len(cfg.TRAIN.SCALES), size=num_images)
    assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
      'num_images ({}) must divide BATCH_SIZE ({})'. \
      format(num_images, cfg.TRAIN.BATCH_SIZE)

    # Get the input image blob, formatted for caffe
    im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)
    #Contains actual image
    blobs = {'data': im_blob}

    assert len(im_scales) == 1, "Single batch only"
    assert len(roidb) == 1, "Single batch only"

    # gt boxes: (x1, y1, x2, y2, cls)
    gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
    dc_len  = roidb[0]['boxes_dc'].shape[0]

    gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
    gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
    gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
    blobs['gt_boxes'] = gt_boxes
    gt_boxes_dc = np.empty((dc_len, 5), dtype=np.float32)
    if cfg.TRAIN.IGNORE_DC:
        gt_ind_dc = np.arange(dc_len)
        gt_boxes_dc[:, 0:4] = roidb[0]['boxes_dc'][gt_ind_dc, :] * im_scales[0]
        gt_boxes_dc[:, 4] = np.zeros(dc_len)
    blobs['gt_boxes_dc'] = gt_boxes_dc
    blobs['im_info'] = np.array(
        [im_blob.shape[1], im_blob.shape[2], im_scales[0]], dtype=np.float32)
    #print('gt boxes')
    assert(len(blobs['gt_boxes']) != 0), 'gt_boxes is empty for image {:s}'.format(roidb[0]['image'])
    return blobs


def _get_image_blob(roidb, scale_inds):
    """Builds an input blob from the images in the roidb at the specified
  scales.
  """
    num_images = len(roidb)
    processed_ims = []
    im_scales = []
    for i in range(num_images):
        im = cv2.imread(roidb[i]['image'])
        #print(roidb[i]['image'])
        #if('000318' in roidb[i]['image']):
        #    print('--------------------------')
        #    print('minibatch images')
        #    print('--------------------------')
        #print(im)
        #print('input image shape')
        #print(im.shape)
        #plt.imshow(im)
        #plt.show()
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, cfg.PIXEL_STDDEVS, target_size,
                                        cfg.TRAIN.MAX_SIZE)
        im_scales.append(im_scale)
        processed_ims.append(im)
        #plt.imshow(im)
        #plt.show()
        #print('reshaped image shape')
        #print(im.shape)
    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, im_scales
