# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick, Sean Bell and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
from model.config import cfg
from model.bbox_transform import bbox_transform, lidar_bbox_transform
from utils.bbox import bbox_overlaps

import torch


def proposal_target_layer(rpn_rois, rpn_scores, gt_boxes, true_gt_boxes, gt_boxes_dc, _num_classes):
    """
  Assign object detection proposals to ground-truth targets. Produces proposal
  classification labels and bounding-box regression targets.
  """

    # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
    # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
    all_rois = rpn_rois
    all_scores = rpn_scores

    # Include ground-truth boxes in the set of candidate rois
    if cfg.TRAIN.USE_GT:
        zeros = rpn_rois.new_zeros(gt_boxes.shape[0], 1)
        all_rois = torch.cat((all_rois, torch.cat(
            (zeros, gt_boxes[:, :-1]), 1)), 0)
        # not sure if it a wise appending, but anyway i am not using it
        all_scores = torch.cat((all_scores, zeros), 0)

    num_images = 1
    rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
    fg_rois_per_image = int(round(cfg.TRAIN.FG_FRACTION * rois_per_image))
    num_bbox_target_elem = true_gt_boxes.shape[1] - 1
    # Sample rois with classification labels and bounding box regression
    # targets
    labels, rois, roi_scores, bbox_targets, bbox_inside_weights = _sample_rois(
        all_rois, all_scores, gt_boxes, true_gt_boxes, gt_boxes_dc, fg_rois_per_image, rois_per_image,
        _num_classes, num_bbox_target_elem)


    rois = rois.view(-1, 5)
    roi_scores = roi_scores.view(-1)
    labels = labels.view(-1, 1)
    bbox_targets = bbox_targets
    bbox_inside_weights = bbox_inside_weights
    bbox_outside_weights = (bbox_inside_weights > 0).float()

    return labels, rois, roi_scores, bbox_targets, bbox_inside_weights, bbox_outside_weights


def _get_bbox_regression_labels(bbox_target_data, num_classes, num_bbox_elem):
    """Bounding-box regression targets (bbox_target_data) are stored in a
  compact form N x (class, tx, ty, tz, tl, tw, th, try)

  This function expands those targets into the 7-of-7*K representation used
  by the network (i.e. only one class has non-zero targets).

  Returns:
      bbox_target (ndarray): N x 7K blob of regression targets
      bbox_inside_weights (ndarray): N x 7K blob of loss weights
  """
    # Inputs are tensor
    clss = bbox_target_data[:, 0]
    bbox_targets = clss.new_zeros(clss.numel(), num_bbox_elem * num_classes)
    bbox_inside_weights = clss.new_zeros(bbox_targets.shape)
    inds = (clss > 0).nonzero().view(-1)
    #numel -> number of elements
    if inds.numel() > 0:
        clss = clss[inds].contiguous().view(-1, 1)
        dim1_inds = inds.unsqueeze(1).expand(inds.size(0), num_bbox_elem)
        #TODO: Very hacky, try to fix.
        base_index = num_bbox_elem * clss
        dim2_list = []
        for i in range(num_bbox_elem):
            dim2_list.append(base_index+i)

        dim2_inds = torch.cat(dim2_list, dim=1).long()
        #dim2_inds_old = torch.cat([num_bbox_elem * clss, num_bbox_elem * clss + 1, num_bbox_elem * clss + 2, num_bbox_elem * clss + 3, num_bbox_elem * clss + 4, num_bbox_elem * clss + 5, num_bbox_elem * clss + 6], 1).long()
        #if(num_bbox_elem == 7):
        #    dim2_inds = torch.cat([num_bbox_elem * clss, num_bbox_elem * clss + 1, num_bbox_elem * clss + 2, num_bbox_elem * clss + 3, num_bbox_elem * clss + 4, num_bbox_elem * clss + 5, num_bbox_elem * clss + 6], 1).long()
        #elif(num_bbox_elem == 4):
        #    dim2_inds = torch.cat([num_bbox_elem * clss, num_bbox_elem * clss + 1, num_bbox_elem * clss + 2, num_bbox_elem * clss + 3], 1).long()
        #else:
        #    print('ERROR: Invalid dim2_inds specified in get_bbox_regression_labels')
        bbox_targets[dim1_inds, dim2_inds] = bbox_target_data[inds][:, 1:]
        bbox_inside_weights[dim1_inds, dim2_inds] = 1.0
        
        #bbox_targets.new(cfg.TRAIN.BBOX_INSIDE_WEIGHTS).view(-1, num_bbox_elem).expand_as(dim1_inds)

    return bbox_targets, bbox_inside_weights

def _get_image_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets (bbox_target_data) are stored in a
  compact form N x (class, tx, ty, tw, th)

  This function expands those targets into the 4-of-4*K representation used
  by the network (i.e. only one class has non-zero targets).

  Returns:
      bbox_target (ndarray): N x 4K blob of regression targets
      bbox_inside_weights (ndarray): N x 4K blob of loss weights
  """
    # Inputs are tensor

    clss = bbox_target_data[:, 0]
    bbox_targets = clss.new_zeros(clss.numel(), 4 * num_classes)
    bbox_inside_weights = clss.new_zeros(bbox_targets.shape)
    inds = (clss > 0).nonzero().view(-1)
    #numel -> number of elements
    if inds.numel() > 0:
        clss = clss[inds].contiguous().view(-1, 1)
        dim1_inds = inds.unsqueeze(1).expand(inds.size(0), 4)
        dim2_inds = torch.cat(
            [4 * clss, 4 * clss + 1, 4 * clss + 2, 4 * clss + 3], 1).long()
        bbox_targets[dim1_inds, dim2_inds] = bbox_target_data[inds][:, 1:]
        bbox_inside_weights[dim1_inds, dim2_inds] = bbox_targets.new(
            cfg.TRAIN.BBOX_INSIDE_WEIGHTS).view(-1, 4).expand_as(dim1_inds)

    return bbox_targets, bbox_inside_weights

def _compute_lidar_targets(ex_rois, roi_height, roi_zc, gt_rois, labels):
    """Compute bounding-box regression targets for a 3d bbox."""
    # Inputs are tensor

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 7

    targets = lidar_bbox_transform(ex_rois, roi_height, roi_zc, gt_rois)
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
        targets = ((targets - targets.new(cfg.LIDAR.BBOX_NORMALIZE_MEANS)) /
                   targets.new(cfg.LIDAR.BBOX_NORMALIZE_STDS))
    return torch.cat([labels.unsqueeze(1), targets], 1)

#ex_rois are pre-computed proposal ROI's to be compared against the GT_ROI's
def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""
    # Inputs are tensor

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = bbox_transform(ex_rois, gt_rois)
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
        targets = ((targets - targets.new(cfg.TRAIN.BBOX_NORMALIZE_MEANS)) /
                   targets.new(cfg.TRAIN.BBOX_NORMALIZE_STDS))
    return torch.cat([labels.unsqueeze(1), targets], 1)


def _sample_rois(all_rois, all_scores, gt_boxes, true_gt_boxes, gt_boxes_dc, fg_rois_per_image,
                 rois_per_image, num_classes, num_bbox_target_elem):
    """Generate a random sample of RoIs comprising foreground and background
  examples. This will provide the 'best-case scenario' for the proposal layer to act as a target 
  """
    # overlaps: (rois x gt_boxes)
    #print('gt boxes')
    #print(gt_boxes)
    max_overlaps_dc = torch.tensor([])
    #Remove all indices that cover dc areas
    if(cfg.TRAIN.IGNORE_DC and list(gt_boxes_dc.size())[0] > 0):
        overlaps_dc = bbox_overlaps(all_rois[:, 1:5].data, gt_boxes_dc[:, :4].data)  #NxK Output N= num roi's k = num gt entries on image
        max_overlaps_dc, _ = overlaps_dc.max(1)  #Returns max value of all input elements along dimension and their index
        dc_inds = (max_overlaps_dc < cfg.TRAIN.DC_THRESH).nonzero().view(-1)
        dc_filtered_rois = all_rois[dc_inds, :]
        dc_filtered_scores = all_scores[dc_inds, :]
    else:
        dc_filtered_rois = all_rois
        dc_filtered_scores = all_scores
    overlaps = bbox_overlaps(dc_filtered_rois[:, 1:5].data, gt_boxes[:, :4].data) #NxK Output N= num roi's k = num gt entries on image
    max_overlaps, gt_assignment = overlaps.max(1) #Returns max value of all input elements along dimension and their index
    labels = gt_boxes[gt_assignment, [4]] #Contains which gt box each overlap is assigned to and the class it belongs to as well
    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = (max_overlaps >= cfg.TRAIN.FG_THRESH).nonzero().view(-1)
    # Guard against the case when an image has fewer than fg_rois_per_image
    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = ((max_overlaps < cfg.TRAIN.BG_THRESH_HI) + (
        max_overlaps >= cfg.TRAIN.BG_THRESH_LO) == 2).nonzero().view(-1) #.nonzero() returns all elements that are non zero in array
    # Small modification to the original version where we ensure a fixed number of regions are sampled
    if fg_inds.numel() > 0 and bg_inds.numel() > 0: #numel() returns number of elements in tensor
        fg_rois_per_image = min(fg_rois_per_image, fg_inds.numel())
        #TODO: This is slow as fuck
        fg_inds = fg_inds[torch.from_numpy( #Randomize order, trim out extras if need be
            npr.choice(
                np.arange(0, fg_inds.numel()),
                size=int(fg_rois_per_image),
                replace=False)).long().to(gt_boxes.device)]
        bg_rois_per_image = rois_per_image - fg_rois_per_image
        to_replace = bg_inds.numel() < bg_rois_per_image #Multiple entries of the same bg inds if too small
        bg_inds = bg_inds[torch.from_numpy(
            npr.choice(
                np.arange(0, bg_inds.numel()),
                size=int(bg_rois_per_image),
                replace=to_replace)).long().to(gt_boxes.device)]
    elif fg_inds.numel() > 0: #Only foreground ROI's were generated
        to_replace = fg_inds.numel() < rois_per_image
        fg_inds = fg_inds[torch.from_numpy(
            npr.choice(
                np.arange(0, fg_inds.numel()),
                size=int(rois_per_image),
                replace=to_replace)).long().to(gt_boxes.device)]
        fg_rois_per_image = rois_per_image
    elif bg_inds.numel() > 0: #Only background ROI's were generated
        to_replace = bg_inds.numel() < rois_per_image
        bg_inds = bg_inds[torch.from_numpy(
            npr.choice(
                np.arange(0, bg_inds.numel()),
                size=int(rois_per_image),
                replace=to_replace)).long().to(gt_boxes.device)]
        fg_rois_per_image = 0
    else:
        print('importing pdb')
        import pdb
        pdb.set_trace()

    # The indices that we're selecting (both fg and bg)
    keep_inds = torch.cat([fg_inds, bg_inds], 0)
    # Select sampled values from various arrays:
    labels = labels[keep_inds].contiguous()
    # Clamp labels for the background RoIs to 0
    labels[int(fg_rois_per_image):] = 0
    rois = dc_filtered_rois[keep_inds].contiguous()
    roi_scores = dc_filtered_scores[keep_inds].contiguous()

    #Right here, bbox_target_data is actually the delta.
    if(cfg.NET_TYPE == 'lidar'):
        #TODO: Multiple anchors??
        roi_height = cfg.LIDAR.ANCHORS[0][2]
        roi_zc     = cfg.LIDAR.ANCHORS[0][2]/2
        bbox_target_data = _compute_lidar_targets(
            rois[:, 1:5].data, roi_height, roi_zc, true_gt_boxes[gt_assignment[keep_inds]][:, :-1].data,
            labels.data)

    elif(cfg.NET_TYPE == 'image'):
        bbox_target_data = _compute_targets(
            rois[:, 1:5].data, gt_boxes[gt_assignment[keep_inds]][:, :4].data,
            labels.data)

    bbox_targets, bbox_inside_weights = \
        _get_bbox_regression_labels(bbox_target_data, num_classes, num_bbox_target_elem)

    return labels, rois, roi_scores, bbox_targets, bbox_inside_weights
