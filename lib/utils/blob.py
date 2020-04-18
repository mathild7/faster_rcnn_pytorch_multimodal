# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
"""Blob helper functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2


def im_list_to_blob(ims):
    """Convert a list of images into a network input.

  Assumes images are already prepared (means subtracted, BGR order, ...).
  """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
    for i in range(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im

    return blob


def prep_im_for_blob(im, pixel_means, pixel_stddev, pixel_arrange, im_scale):
    """Mean subtract and scale an image for use in a blob."""
    im_shape = im.shape
    #im_size_min = np.min(im_shape[0:2])
    #im_size_max = np.max(im_shape[0:2])
    #im_scale = float(target_size) / float(im_size_min)

    im = im.astype(np.float32, copy=False)

    im = cv2.resize(
        im,
        None,
        None,
        fx=im_scale,
        fy=im_scale,
        interpolation=cv2.INTER_LINEAR)
    im = im.astype(np.float32)
    im = im[:,:,pixel_arrange]
    im -= pixel_means
    im = im/pixel_stddev
    # Prevent the biggest axis from being more than MAX_SIZE

    return im


def bev_map_list_to_blob(bev_maps):
    """Convert a list of images into a network input.

  Assumes images are already prepared (means subtracted, BGR order, ...).
  """
    max_shape = np.array([bm.shape for bm in bev_maps]).max(axis=0)
    num_bev_maps = len(bev_maps)
    blob = np.zeros((num_bev_maps, max_shape[0], max_shape[1], max_shape[2]),
                    dtype=np.float32)
    for i in range(num_bev_maps):
        bev_map = bev_maps[i]
        blob[i, 0:bev_map.shape[0], 0:bev_map.shape[1], 0:bev_map.shape[2]] = bev_map

    return blob


def prep_bev_map_for_blob(bev_map, means, vars, scale):
    bev_map_resize = np.resize(bev_map,bev_map.shape*scale)
    bev_map_resize = bev_map_resize/np.sqrt(vars)
    bev_map_resize = bev_map_resize-means
    return bev_map_resize
