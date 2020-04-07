# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import os
import shutil
import math

from model.config import cfg, get_output_dir
from model.bbox_transform import clip_boxes, bbox_transform_inv
import utils.bbox as bbox_utils
import torch
from torchvision.ops import nms

#TODO: Rework to accept all kinds of variance
def nms_hstack_var_torch(var_type,var,inds,keep,c,bbox_elem):
    if(var_type == 'cls'):
        cls_var = var[inds]
        #Removed dependency on class, as entropy is measured across all classes
        #cls_var = var[inds,c]
        cls_var = cls_var[keep].unsqueeze(1)
        return cls_var.cpu().numpy()
    elif(var_type == 'cls_var'):
        cls_var = var[inds,c]
        #Removed dependency on class, as entropy is measured across all classes
        #cls_var = var[inds,c]
        cls_var = cls_var[keep].unsqueeze(1)
        return cls_var.cpu().numpy()
    elif(var_type == 'bbox'):
        cls_bbox_var = var[inds, c * bbox_elem:(c + 1) * bbox_elem]
        cls_bbox_var = cls_bbox_var[keep, :].cpu().numpy()
        #bbox_samples = samples[inds, c * 4:(c + 1) * 4]
        #bbox_samples = bbox_samples[keep, :, :]
        return cls_bbox_var
    else:
        return None

def nms_hstack_torch(scores,mean_boxes,thresh,c,bbox_elem,db_type):
    inds = torch.where(scores[:, c] > thresh)[0]
    #inds         = np.where(scores[:, c] > thresh)[0]
    #No detections over threshold
    if(inds.shape[0] == 0):
        print('no detections for image over threshold {}'.format(thresh))
        return np.empty(0),[],[]
    cls_scores   = scores[inds, c]
    cls_boxes    = mean_boxes[inds, c * bbox_elem:(c + 1) * bbox_elem]
    if(db_type == 'lidar'):
        cls_boxes_aabb = bbox_utils.bbaa_graphics_gems_torch(cls_boxes, 0,  0, clip=False)
    #[cls_var,cls_boxes,cls_scores]
    cls_dets = np.hstack((cls_boxes.cpu().numpy(), cls_scores.unsqueeze(1).cpu().numpy())) \
        .astype(np.float32, copy=False)
    if(db_type == 'lidar'):
        keep = nms(cls_boxes_aabb, cls_scores, cfg.TEST.NMS).cpu().numpy() if cls_dets.size > 0 else []
    else:
        keep = nms(cls_boxes, cls_scores, cfg.TEST.NMS).cpu().numpy() if cls_dets.size > 0 else []
    cls_dets = cls_dets[keep, :]
    #Only if this variable has been provided
    return cls_dets, inds, keep



#TODO: Could use original imwidth/imheight
def filter_and_draw_prep(rois, cls_score, pred_boxes, uncertainties, info, num_classes,thresh=0.1,db_type='none'):
    #print('validation img properties h: {} w: {} s: {} '.format(imheight,imwidth,imscale))
    frame_width = info[1] - info[0]
    frame_height = info[3] - info[2]
    scale = info[6]
    rois = rois[:, 1:5].detach().cpu().numpy()
    #pred_boxes = bbox_transform_inv(torch.from_numpy(rois), torch.from_numpy(bbox_pred)).numpy()
    if(db_type == 'image'):
        bbox_elem         = 4
        # x1 >= 0
        pred_boxes[:, 0::bbox_elem] = torch.clamp_min(pred_boxes[:, 0::bbox_elem],0)
        # y1 >= 0
        pred_boxes[:, 1::bbox_elem] = torch.clamp_min(pred_boxes[:, 1::bbox_elem], 0)
        # x2 < imwidth
        pred_boxes[:, 2::bbox_elem] = torch.clamp_max(pred_boxes[:, 2::bbox_elem],frame_width/scale - 1)
        # y2 < imheight 3::4 means start at 3 then jump every 4
        pred_boxes[:, 3::bbox_elem] = torch.clamp_max(pred_boxes[:, 3::bbox_elem],frame_height/scale - 1)
    elif(db_type == 'lidar'):
        bbox_elem         = 7
    else:
        return None
        
    all_boxes = []
    all_uncertainties = []
    #print('----------------')
    #print(pred_boxes)
    torch.set_printoptions(profile="default")
    # skip j = 0, because it's the background class
    #for i,score in enumerate(cls_score):
    #    for j,cls_s in enumerate(score):
    #        if((cls_s > thresh or cls_s < 0.0) and j > 0):
                #print('score for entry {} and class {} is {}'.format(i,j,cls_s))
    all_boxes       = [[] for _ in range(num_classes)]
    all_uncertainty = [{} for _ in range(num_classes)]
    for j in range(1, num_classes):
        #Don't need to stack variance here, it is only used in trainval to draw stuff
        all_box, inds, keep = nms_hstack_torch(cls_score,pred_boxes,thresh,j,bbox_elem,db_type)
        uncertainties = {}
        if(len(keep) != 0):
            if(cfg.ENABLE_ALEATORIC_BBOX_VAR):
                uncertainties['a_bbox_var'] = nms_hstack_var_torch('bbox',uncertainties['a_bbox_var'],inds,keep,j,bbox_elem)
            else:
                a_bbox_var = np.zeros(bbox_elem)
            if(cfg.ENABLE_EPISTEMIC_BBOX_VAR):
                uncertainties['e_bbox_var'] = nms_hstack_var_torch('bbox',uncertainties['e_bbox_var'],inds,keep,j,bbox_elem)
            else:
                e_bbox_var = np.zeros(bbox_elem)
            if(cfg.ENABLE_ALEATORIC_CLS_VAR):
                #uncertainties['a_cls_entropy'] = a_cls_entropy
                uncertainties['a_cls_entropy'] = nms_hstack_var_torch('cls',uncertainties['a_cls_entropy'],inds,keep,j,bbox_elem)
                uncertainties['a_cls_var']     = nms_hstack_var_torch('cls_var',uncertainties['a_cls_var'],inds,keep,j,bbox_elem)
            else:
                a_cls_var = [0]
            if(cfg.ENABLE_EPISTEMIC_CLS_VAR):
                uncertainties['e_cls_mutual_info'] = nms_hstack_var_torch('cls',uncertainties['e_cls_mutual_info'],inds,keep,j,bbox_elem)
            else:
                e_cls_var = [0]
        all_uncertainty[j] = uncertainties
        all_boxes[j] = all_box
    #torch.set_printoptions(profile="full")
    #print(bbox_pred)

    return rois, all_boxes, all_uncertainty