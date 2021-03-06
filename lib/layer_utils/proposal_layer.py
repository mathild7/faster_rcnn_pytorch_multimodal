# --------------------------------------------------------
# Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from model.config import cfg
from model.bbox_transform import bbox_transform_inv, clip_boxes
from torchvision.ops import nms

import torch


def proposal_layer(rpn_cls_prob, rpn_bbox_pred, info, cfg_key,
                   anchors, anchors_3d, num_anchors):
    """A simplified version compared to fast/er RCNN
     For details please see the technical report
  """
    #print('rpn_bbox_pred')
    #print(rpn_bbox_pred)
    if type(cfg_key) == bytes:
        cfg_key = cfg_key.decode('utf-8')
    pre_nms_topN  = cfg[cfg_key].RPN_PRE_NMS_TOP_N
    post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
    nms_thresh    = cfg[cfg_key].RPN_NMS_THRESH

    # Get the scores and bounding boxes
    scores        = rpn_cls_prob[:, :, :, num_anchors:] #Start at back half of anchor/score list
    rpn_bbox_pred = rpn_bbox_pred.view((-1, 4)) #rpn_bbox_pred are adjustment factors to existing anchors
    scores        = scores.contiguous().view(-1, 1) #Collapse into a single vector
    proposals     = bbox_transform_inv(anchors, rpn_bbox_pred)
    proposals     = clip_boxes(proposals, info) #Make sure they are within bounds

    # Pick the top 'pre_nms_topN' # of region proposals
    scores, order = scores.view(-1).sort(descending=True)
    if pre_nms_topN > 0:
        order  = order[:pre_nms_topN]
        scores = scores[:pre_nms_topN].view(-1, 1)
    proposals  = proposals[order.data, :]
    anchors_3d = anchors_3d[order.data, :]
    # Non-maximal suppression
    keep = nms(proposals, scores.squeeze(1), nms_thresh)
    # Pick th top region proposals after NMS
    if post_nms_topN > 0:
        keep = keep[:post_nms_topN]
    proposals = proposals[keep, :]
    scores    = scores[keep, ]
    anchors_3d = anchors_3d[keep, :]
    # Only support single image as input
    batch_inds = proposals.new_zeros(proposals.size(0), 1)
    blob       = torch.cat((batch_inds, proposals), 1)

    return blob, scores, anchors_3d
