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


def prep_im_for_blob(im, pixel_means, pixel_stddev, pixel_arrange, target_size, max_size):
    """Mean subtract and scale an image for use in a blob."""
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)

    im = im.astype(np.float32, copy=False)

    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
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

    return im, im_scale


def voxel_grid_list_to_blob(voxel_grids):
    """Convert a list of images into a network input.

  Assumes images are already prepared (means subtracted, BGR order, ...).
  """
    max_shape = np.array([voxel_grids.shape for vg in voxel_grids]).max(axis=0)
    num_voxel_grids = len(voxel_grids)
    blob = np.zeros((num_voxel_grids, max_shape[0], max_shape[1], max_shape[2]),
                    dtype=np.float32)
    for i in range(num_voxel_grids):
        voxel_grid = voxel_grids[i]
        blob[i, 0:voxel_grid.shape[0], 0:voxel_grid.shape[1], 0:voxel_grid.shape[2]] = voxel_grid

    return blob


def prep_voxel_grid_for_blob(voxel_grid, means, vars, scale):
    voxel_grid = np.resize(voxel_grid,voxel_grid.shape*scale)
    voxel_grid = voxel_grid/np.sqrt(vars)
    voxel_grid = voxel_grid-means
    # Prevent the biggest axis from being more than MAX_SIZE

    return voxel_grid
