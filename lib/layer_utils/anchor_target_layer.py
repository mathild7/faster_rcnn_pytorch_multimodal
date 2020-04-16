# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from model.config import cfg
import numpy as np
import numpy.random as npr
from utils.bbox import bbox_overlaps, bbox_overlaps_3d
from model.bbox_transform import bbox_transform
import torch


#From generated anchor boxes, select subset that have a large overlap with GT_Boxes

def anchor_target_layer_torch(gt_boxes, gt_boxes_dc, info, _feat_stride,
                              all_anchors, num_anchors, height, width, dev):
    """Same as the anchor target layer in original Fast/er RCNN """
    A = num_anchors
    #print('num anchors')
    #print(num_anchors)
    #print(im_info[1])
    #print(im_info[0])
    total_anchors = all_anchors.shape[0]
    K = total_anchors / num_anchors

    # allow boxes to sit over the edge by a small amount
    _allowed_border = 0

    # only keep anchors inside the image
    inds_inside = torch.where(
        (all_anchors[:, 0] >= info[0] - _allowed_border) &  #width_max
        (all_anchors[:, 1] >= info[2] - _allowed_border) &  #height_min
        (all_anchors[:, 2] < info[1]  + _allowed_border) &  # width_max
        (all_anchors[:, 3] < info[3]  + _allowed_border)  # height_max
    )[0]

    # keep only inside anchors
    anchors = all_anchors[inds_inside, :]

    # label: 1 is positive, 0 is negative, -1 is dont care
    #Subset of anchors within image boundary
    labels = torch.full((len(inds_inside), ), -1, dtype=torch.int64).to(device=dev)
    #labels.fill(-1)

    # overlaps between the anchors and the gt boxes
    # overlaps (ex, gt)
    #from utils.bbox import bbox_overlaps
    overlaps = bbox_overlaps(
        anchors.contiguous(),
        gt_boxes.contiguous())
    if cfg.TRAIN.IGNORE_DC:
        overlaps_dc = bbox_overlaps(
            anchors.contiguous(),
            gt_boxes_dc.contiguous())
        overlaps_dc_idx = torch.argwhere(overlaps_dc > cfg.TRAIN.DC_THRESH)
        labels[overlaps_dc_idx[:, 0]] = -1
    #overlaps: (N, K) overlap between boxes and query_boxes
    argmax_overlaps = overlaps.argmax(dim=1)
    #grab subset of 2D array to only get [:,max_overlap_index] 
    max_overlaps = overlaps[torch.arange(len(inds_inside)).to(device=dev), argmax_overlaps]
    #max_overlaps_2 = torch.index_select(overlaps, 0, argmax_overlaps)
    gt_argmax_overlaps = overlaps.argmax(dim=0)
    #grab same subset of 2D array to get corresponding GT boxes with their max overlap counterpart
    gt_max_overlaps = overlaps[gt_argmax_overlaps, torch.arange(overlaps.shape[1]).to(device=dev)]
    gt_argmax_overlaps = torch.where(overlaps == gt_max_overlaps)[0]

    if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
        # assign bg labels first so that positive labels can clobber them
        # first set the negatives
        labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

    # fg label: for each gt, anchor with highest overlap
    #gt_argmax_overlaps is an index subset of the anchors that max overlap with a gt box
    labels[gt_argmax_overlaps] = 1

    # fg label: above threshold IOU
    #anything else needs a large overlap as well
    labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1
    if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
        # assign bg labels last so that negative labels can clobber positives
        labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
    # subsample positive labels if we have too many
    num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
    fg_inds = torch.where(labels == 1)[0]
    #TODO: Really, randomly select indices to disable? Why not worst ones? At least dont do this for the argmax..
    #If too many foreground entries
    if len(fg_inds) > num_fg:
        perm = torch.randperm(fg_inds.numel(), device=dev)[num_fg:]
        fg_inds_subset = fg_inds[perm]
        labels[fg_inds_subset] = -1

    # subsample negative labels if we have too many
    fg_sum = torch.sum(labels == 1)
    num_bg = cfg.TRAIN.RPN_BATCHSIZE - fg_sum
    bg_inds = torch.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        perm = torch.randperm(bg_inds.numel(), device=dev)[num_bg:]
        bg_inds_subset = bg_inds[perm]
        labels[bg_inds_subset] = -1
    #Find target bounding boxes
    #bbox_targets = torch.zeros((len(inds_inside), 4), dtype=torch.float32).to(device=dev)
    bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])
    #print('GT BOXES')
    #print(bbox_targets.shape)
    bbox_inside_weights = torch.zeros((len(inds_inside), 4), dtype=torch.float32).to(device=dev)
    # only the positive ones have regression targets
    bbox_inside_weights[labels == 1, :] = torch.from_numpy(np.array(
        cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS, dtype=np.float32)).to(device=dev)
    bbox_outside_weights = torch.zeros((len(inds_inside), 4), dtype=torch.float32).to(device=dev)
    if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
        # uniform weighting of examples (given non-uniform sampling) num_examples is a max of 256 by default
        num_examples = torch.sum(labels >= 0)
        #positive_weights = torch.ones((1, 4)) * 1.0 / num_examples
        #negative_weights = torch.ones((1, 4)) * 1.0 / num_examples
        positive_weights = 1.0/float(num_examples)
        negative_weights = 1.0/float(num_examples)
    else:
        assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
        #TODO: Broken
        positive_weights = (cfg.TRAIN.RPN_POSITIVE_WEIGHT / torch.sum(labels == 1))
        negative_weights = ((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) / torch.sum(labels == 0))
    bbox_outside_weights[labels == 1, :] = positive_weights
    bbox_outside_weights[labels == 0, :] = negative_weights
    #print('bbox weights')
    #print(bbox_outside_weights)
    #print(bbox_inside_weights)
    # map up to original set of anchors
    labels = _unmap(labels.type(dtype=torch.float32), total_anchors, inds_inside, fill=-1, dev=dev)
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0, dev=dev)
    bbox_inside_weights = _unmap(
        bbox_inside_weights, total_anchors, inds_inside, fill=0, dev=dev)
    bbox_outside_weights = _unmap(
        bbox_outside_weights, total_anchors, inds_inside, fill=0, dev=dev)

    # labels
    labels = labels.reshape((1, height, width, A)).permute(0, 3, 1, 2)
    #labels = labels.reshape((1, 1, A * height, width))
    rpn_labels = labels

    # bbox_targets
    bbox_targets = bbox_targets \
      .reshape((1, height, width, A * 4))

    rpn_bbox_targets = bbox_targets
    # bbox_inside_weights
    bbox_inside_weights = bbox_inside_weights \
      .reshape((1, height, width, A * 4))

    rpn_bbox_inside_weights = bbox_inside_weights

    # bbox_outside_weights
    bbox_outside_weights = bbox_outside_weights \
      .reshape((1, height, width, A * 4))

    rpn_bbox_outside_weights = bbox_outside_weights
    return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights



#From generated anchor boxes, select subset that have a large overlap with GT_Boxes
#Info is now [min_x, max_x, min_y, max_y, scale]
def anchor_target_layer(gt_boxes, gt_boxes_dc, info, _feat_stride,
                        all_anchors, num_anchors, height, width):
    """Same as the anchor target layer in original Fast/er RCNN """
    A = num_anchors
    #print('num anchors')
    #print(num_anchors)
    #print(info[1])
    #print(info[0])
    total_anchors = all_anchors.shape[0]
    K = total_anchors / num_anchors

    # allow boxes to sit over the edge by a small amount
    _allowed_border = 0

    # only keep anchors inside the frame
    #TODO: Torchify

    #TODO: Subtract minimum value between GT boxes and anchors as to not get the overlaps issue (maybe also track and see it happen?)

    inds_inside = np.where(
        (all_anchors[:, 0] >= info[0] - _allowed_border) &  #width_max
        (all_anchors[:, 1] >= info[2] - _allowed_border) &  #height_min
        (all_anchors[:, 2] < info[1]  + _allowed_border) &  # width_max
        (all_anchors[:, 3] < info[3]  + _allowed_border)  # height_max
    )[0]

    # keep only inside anchors
    anchors = all_anchors[inds_inside, :]

    # label: 1 is positive, 0 is negative, -1 is dont care
    #Subset of anchors within image boundary
    labels = np.empty((len(inds_inside), ), dtype=np.float32)
    labels.fill(-1)

    # overlaps between the anchors and the gt boxes
    # overlaps (ex, gt)
    #from utils.bbox import bbox_overlaps
    overlaps = bbox_overlaps(
        np.ascontiguousarray(anchors, dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float))
    #np.set_printoptions(threshold=np.inf)
    #print('----------------------------------------------')
    #overlaps_trimmed = overlaps[~np.all(overlaps == 0, axis=1)]
    #print(overlaps_trimmed)
    #print('----------------------------------------------')
    if cfg.TRAIN.IGNORE_DC:
        overlaps_dc = bbox_overlaps(
            np.ascontiguousarray(anchors, dtype=np.float),
            np.ascontiguousarray(gt_boxes_dc, dtype=np.float))
        overlaps_dc_idx = np.argwhere(overlaps_dc > cfg.TRAIN.DC_THRESH)
        labels[overlaps_dc_idx[:, 0]] = -1
    #overlaps: (N, K) overlap between boxes and query_boxes
    argmax_overlaps = overlaps.argmax(axis=1) #Best fiting GT for each anchor (1,N)
    gt_argmax_overlaps = overlaps.argmax(axis=0) #Best fitting anchor for each GT box (K,1)
    #grab subset of 2D array to only get [:,max_overlap_index] 
    #max_overlaps = overlaps.take(argmax_overlaps,axis=1)
    #np.set_printoptions(threshold=np.inf)
    #print(argmax_overlaps)
    #print(overlaps)
    max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
    #max_overlaps = overlaps[:, argmax_overlaps]
    #max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
    #grab same subset of 2D array to get corresponding GT boxes with their max overlap counterpart
    #gt_max_overlaps = overlaps[gt_argmax_overlaps,
    #                           np.arange(overlaps.shape[1])]
    #TODO: How the fuck does this work
    #gt_max_overlaps = overlaps[gt_argmax_overlaps,:]
    gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]
    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

    if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
        # assign bg labels first so that positive labels can clobber them
        # first set the negatives
        labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

    # fg label: for each gt, anchor with highest overlap
    #gt_argmax_overlaps is an index subset of the anchors that max overlap with a gt box
    labels[gt_argmax_overlaps] = 1

    # fg label: above threshold IOU
    #anything else needs a large overlap as well
    #TODO: Distance based overlap threshold?
    labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

    if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
        # assign bg labels last so that negative labels can clobber positives
        labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
    # subsample positive labels if we have too many
    num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
    fg_inds = np.where(labels == 1)[0]
    #TODO: Really, randomly select indices to disable? Why not worst ones? At least dont do this for the argmax..
    #If too many foreground entries
    if len(fg_inds) > num_fg:
        disable_inds = npr.choice(
            fg_inds, size=(len(fg_inds) - num_fg), replace=False)
        labels[disable_inds] = -1

    # subsample negative labels if we have too many
    num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = npr.choice(
            bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        labels[disable_inds] = -1
    #Find target bounding boxes
    bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
    bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])
    #print('GT BOXES')
    #print(bbox_targets.shape)
    bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    #Create a mask where labels == 1
    bbox_inside_weights[labels == 1, :] = np.array(
        cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS)

    bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    #Sample weighting is turned off
    if int(cfg.TRAIN.RPN_POSITIVE_WEIGHT) == -1:
        # uniform weighting of examples (given non-uniform sampling) num_examples is a max of 256 by default
        num_examples = np.sum(labels >= 0)
        positive_weights = np.ones((1, 4)) * 1.0 / num_examples
        negative_weights = np.ones((1, 4)) * 1.0 / num_examples
    else:
        assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
        positive_weights = (cfg.TRAIN.RPN_POSITIVE_WEIGHT / np.sum(labels == 1))
        negative_weights = ((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) / np.sum(labels == 0))
    
    bbox_outside_weights[labels == 1, :] = positive_weights
    bbox_outside_weights[labels == 0, :] = negative_weights
    #print('bbox weights')
    #print(bbox_outside_weights)
    #print(bbox_inside_weights)
    # map up to original set of anchors
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
    bbox_inside_weights = _unmap(
        bbox_inside_weights, total_anchors, inds_inside, fill=0)
    bbox_outside_weights = _unmap(
        bbox_outside_weights, total_anchors, inds_inside, fill=0)

    # labels
    labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
    #labels = labels.reshape((1, 1, A * height, width))
    rpn_labels = labels

    # bbox_targets
    bbox_targets = bbox_targets \
      .reshape((1, height, width, A * 4))

    rpn_bbox_targets = bbox_targets
    # bbox_inside_weights
    bbox_inside_weights = bbox_inside_weights \
      .reshape((1, height, width, A * 4))

    rpn_bbox_inside_weights = bbox_inside_weights

    # bbox_outside_weights
    bbox_outside_weights = bbox_outside_weights \
      .reshape((1, height, width, A * 4))

    rpn_bbox_outside_weights = bbox_outside_weights
    return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights


def _unmap(data, count, inds, fill=0,dev=None,datatype=torch.float32):
    """ Unmap a subset of item (data) back to the original set of items (of
  size count) """
    if isinstance(data, np.ndarray):
        #1D array
        if len(data.shape) == 1:
            ret = np.empty((count, ), dtype=np.float32)
            ret.fill(fill)
            ret[inds] = data
        #ND array
        else:
            ret = np.empty((count, ) + data.shape[1:], dtype=np.float32)
            ret.fill(fill)
            ret[inds, :] = data
    else:
        #1D array
        if len(data.shape) == 1:
            ret = torch.full((count, ), fill, dtype=torch.float32).to(device=dev)
            ret[inds] = data
        #ND array
        else:
            ret = torch.full((count, ) + data.shape[1:], fill, dtype=torch.float32).to(device=dev)
            ret[inds, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 5
    if isinstance(ex_rois, np.ndarray):
        ex_rois = torch.from_numpy(ex_rois)
    if isinstance(gt_rois, np.ndarray):
        gt_rois = torch.from_numpy(gt_rois)
    return bbox_transform(ex_rois, gt_rois[:, :4])