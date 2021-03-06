# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import math
from model.config import cfg

def lidar_3d_bbox_transform(ex_rois, ex_anchors, gt_rois):
    
    roi_lengths  = (ex_rois[:,2] - ex_rois[:,0] + 1)
    roi_widths   = (ex_rois[:,3] - ex_rois[:,1] + 1)
    ex_lengths  = roi_lengths  #ex_anchors[:,3]
    ex_widths   = roi_widths  #ex_anchors[:,4]
    ex_heights  = ex_anchors[:,5]
    gt_heights  = gt_rois[:,5]
    gt_z_ctr    = gt_rois[:,2]

    ex_ctr_x    = ex_rois[:,0] + roi_lengths/2.0
    ex_ctr_y    = ex_rois[:,1] + roi_widths/2.0
    ex_ctr_z    = ex_anchors[:,2]
    #TODO: TEST!!
    inv_mask    = torch.where(roi_lengths >= roi_widths, torch.tensor([0]).to(device=roi_lengths.device), torch.tensor([1]).to(device=roi_lengths.device))
    inv_mask    = inv_mask.float()
    ex_headings = inv_mask*math.pi/2.0
    ex_areas    = torch.sqrt(torch.pow(ex_lengths,2) + torch.pow(ex_widths,2))
    #ex_headings = ex_anchors[:,6]
    #TODO: change targets to be divided by BEV area
    targets_dx = (gt_rois[:,0] - ex_ctr_x) / ex_areas
    targets_dl = torch.log(gt_rois[:,3] / ex_lengths)

    targets_dy = (gt_rois[:,1] - ex_ctr_y) / ex_areas
    targets_dw = torch.log(gt_rois[:,4] / ex_widths)

    #targets_dz = (gt_rois[:,2] - cfg.LIDAR.Z_RANGE[0] - ex_ctr_z) / ex_heights
    targets_dz = (gt_rois[:,2] - ex_ctr_z) / ex_heights
    targets_dh = torch.log(gt_rois[:,5] / ex_heights)
    #TODO: Apply [Pi/2,-Pi/2) clipping
    #targets_ry = torch.sin(ex_headings) #- gt_rois[:, 6])
    targets_ry  = gt_rois[:, 6]  #- ex_headings
    targets = torch.stack((targets_dx, targets_dy, targets_dz, targets_dl, targets_dw, targets_dh, targets_ry), 1)
    return targets


def bbox_transform(ex_rois, gt_rois):
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_area   = torch.sqrt(torch.pow(ex_widths,2) + torch.pow(ex_heights,2))
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_area  #ex_area  #ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_area  #ex_area  #ex_heights
    targets_dw = torch.log(gt_widths / ex_widths)
    targets_dh = torch.log(gt_heights / ex_heights)

    targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), 1)
    return targets


#Box defines original ROI and then is adjusted by the regression deltas
#Box (x1,y1,x2,y2)
def bbox_transform_inv(boxes, deltas, scales=None):
    # Input should be both tensor or both Variable and on the same device
    if(scales is not None):
        boxes = boxes/scales
    if len(boxes) == 0:
        return deltas.detach() * 0

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    area    = torch.sqrt(torch.pow(widths,2) + torch.pow(heights,2))
    #Re-centering top left hand corner
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights
    #e.g. 16 elements for 4 classes
    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * area.unsqueeze(1) + ctr_x.unsqueeze(1)  #area.unsqueeze(1) + ctr_x.unsqueeze(1)  #widths.unsqueeze(1) + ctr_x.unsqueeze(1)
    pred_ctr_y = dy * area.unsqueeze(1) + ctr_y.unsqueeze(1)  #area.unsqueeze(1) + ctr_y.unsqueeze(1)  #heights.unsqueeze(1) + ctr_y.unsqueeze(1)
    pred_w = torch.exp(dw) * widths.unsqueeze(1)
    pred_h = torch.exp(dh) * heights.unsqueeze(1)

    pred_boxes = torch.cat(\
      [_.unsqueeze(2) for _ in [pred_ctr_x - 0.5 * pred_w,
                                pred_ctr_y - 0.5 * pred_h,
                                pred_ctr_x + 0.5 * pred_w,
                                pred_ctr_y + 0.5 * pred_h]], 2).view(len(boxes), -1)

    return pred_boxes

def uncertainty_transform_inv(boxes, deltas, uncertainty, scales=None):
    if(scales is not None):
        boxes = boxes/scales
    
    lengths  = (boxes[:,2] - boxes[:,0] + 1)
    widths   = (boxes[:,3] - boxes[:,1] + 1)
    uc_x    = uncertainty[:, 0::7]
    uc_y    = uncertainty[:, 1::7]
    uc_l    = uncertainty[:, 3::7]
    uc_w    = uncertainty[:, 4::7]

    uc_x    = uc_x*lengths
    uc_y    = uc_y*widths
    uc_l    = torch.exp(uc_l)-1
    uc_w    = torch.exp(uc_w)-1

    uc_inv_arr = [_.unsqueeze(2) for _ in [uc_x,
                                           uc_y,
                                           uc_l,
                                           uc_w]]
    uncertainty_inv = torch.cat(uc_inv_arr, 2).view(len(boxes), -1)
    return torch.pow(uncertainty_inv,2)



def lidar_3d_uncertainty_transform_inv(rois, boxes, deltas, uncertainty, scales=None):
    if(scales is not None):
        boxes[:, 0:2] = boxes[:, 0:2]/scales
        boxes[:, 3:5] = boxes[:, 3:5]/scales
        rois = rois/scales
   
    roi_lengths  = (rois[:,2] - rois[:,0] + 1)
    roi_widths   = (rois[:,3] - rois[:,1] + 1)
    lengths = roi_lengths  #boxes[:, 3]
    widths  = roi_widths  #boxes[:, 4]
    heights = boxes[:, 5]

    uc_x    = uncertainty[:, 0::7]
    uc_y    = uncertainty[:, 1::7]
    uc_z    = uncertainty[:, 2::7]
    uc_l    = uncertainty[:, 3::7]
    uc_w    = uncertainty[:, 4::7]
    uc_h    = uncertainty[:, 5::7]
    uc_rot  = uncertainty[:, 6::7]

    uc_x    = uc_x*lengths.unsqueeze(1)
    uc_y    = uc_y*widths.unsqueeze(1)
    uc_z    = uc_z*heights.unsqueeze(1)
    uc_l    = torch.exp(uc_l)-1
    uc_w    = torch.exp(uc_w)-1
    uc_h    = torch.exp(uc_h)-1
    uc_rot  = uc_rot

    uc_inv_arr = [_.unsqueeze(2) for _ in [uc_x,
                                           uc_y,
                                           uc_z,
                                           uc_l,
                                           uc_w,
                                           uc_h,
                                           uc_rot]]
    uncertainty_inv = torch.cat(uc_inv_arr, 2).view(len(boxes), -1)

    return torch.pow(uncertainty_inv,2)




def lidar_3d_bbox_transform_inv(rois, boxes, deltas, scales=None):
    # Input should be both tensor or both Variable and on the same device
    #This is handled by voxel_grid_to_pc()
    if(scales is not None):
        boxes[:, 0:2] = boxes[:, 0:2]/scales
        boxes[:, 3:5] = boxes[:, 3:5]/scales
        rois = rois/scales
    if len(boxes) == 0:
        return deltas.detach() * 0


    roi_lengths  = (rois[:,2] - rois[:,0] + 1)
    roi_widths   = (rois[:,3] - rois[:,1] + 1)

    #TODO: TEST!!
    inv_mask    = torch.where(roi_lengths >= roi_widths, torch.tensor([0]).to(device=roi_lengths.device), torch.tensor([1]).to(device=roi_lengths.device))
    inv_mask    = inv_mask.float()
    lengths = roi_lengths  #boxes[:, 3]
    widths  = roi_widths  #boxes[:, 4]
    heights = boxes[:, 5]
    #heading = boxes[:, 6]
    heading = inv_mask*math.pi/2.0
    #Re-centering top left hand corner

    ctr_x    = rois[:,0] + roi_lengths/2.0
    ctr_y    = rois[:,1] + roi_widths/2.0
    #ctr_x = boxes[:, 0]
    #ctr_y = boxes[:, 1]
    ctr_z = boxes[:, 2]

    #e.g. 16 elements for 4 classes
    dx = deltas[:, 0::7]
    dy = deltas[:, 1::7]
    dz = deltas[:, 2::7]
    dl = deltas[:, 3::7]
    dw = deltas[:, 4::7]
    dh = deltas[:, 5::7]
    dr = deltas[:, 6::7]
    areas = torch.sqrt(torch.pow(lengths,2)+torch.pow(widths,2))
    pred_ctr_x = dx * areas.unsqueeze(1) + ctr_x.unsqueeze(1)
    pred_ctr_y = dy * areas.unsqueeze(1) + ctr_y.unsqueeze(1)
    pred_ctr_z = dz * heights.unsqueeze(1) + ctr_z.unsqueeze(1)  # + cfg.LIDAR.Z_RANGE[0]
    pred_l = torch.exp(dl) * lengths.unsqueeze(1)
    pred_w = torch.exp(dw) * widths.unsqueeze(1)
    pred_h = torch.exp(dh) * heights.unsqueeze(1)
    pred_ry = dr  #+ heading.unsqueeze(1)
    #Lock headings to be [pi/2, -pi/2)
    pi2 = float(math.pi/2.0)
    #pred_ry = torch.where(pred_ry > pi2, pred_ry - math.pi, pred_ry)
    #pred_ry = torch.where(pred_ry <= -pi2, pred_ry + math.pi, pred_ry)

    pred_boxes = torch.cat(
        [_.unsqueeze(2) for _ in [pred_ctr_x,
                                  pred_ctr_y,
                                  pred_ctr_z,
                                  pred_l,
                                  pred_w,
                                  pred_h,
                                  pred_ry]], 2).view(len(boxes), -1)
    return pred_boxes

def clip_boxes(boxes, shape):
    """
  Clip boxes to image boundaries.
  boxes must be tensor or Variable, shape can be anything but Variable
  """

    if not hasattr(boxes, 'data'):
        boxes_ = boxes.numpy()

    boxes = boxes.view(boxes.size(0), -1, 4)
    #TODO: Have i just broken clip boxes? Or was it broken already? It used to be
    #boxes = torch.stack(\
    #  [boxes[:,:,0].clamp(0, shape[0]),
    #   boxes[:,:,1].clamp(0, shape[1]),
    #   boxes[:,:,2].clamp(0, shape[0]),
    #   boxes[:,:,3].clamp(0, shape[1])], 2).view(boxes.size(0), -1)

    boxes = torch.stack([boxes[:,:,0].clamp(shape[0], shape[1] - 1),
                         boxes[:,:,1].clamp(shape[2], shape[3] - 1),
                         boxes[:,:,2].clamp(shape[0], shape[1] - 1),
                         boxes[:,:,3].clamp(shape[2], shape[3] - 1)], 2).view(boxes.size(0), -1)

    return boxes
