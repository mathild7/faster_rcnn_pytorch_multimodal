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

import torch
from torchvision.ops import nms

#TODO: Rework to accept all kinds of variance
def nms_hstack_var_torch(var_type,var,inds,keep,c):
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
        cls_bbox_var = var[inds, c * 4:(c + 1) * 4]
        cls_bbox_var = cls_bbox_var[keep, :].cpu().numpy()
        #bbox_samples = samples[inds, c * 4:(c + 1) * 4]
        #bbox_samples = bbox_samples[keep, :, :]
        return cls_bbox_var
    else:
        return None

def nms_hstack_torch(scores,mean_boxes,thresh,c):
    inds = torch.where(scores[:, c] > thresh)[0]
    #inds         = np.where(scores[:, c] > thresh)[0]
    #No detections over threshold
    if(inds.shape[0] == 0):
        print('no detections for image over threshold {}'.format(thresh))
        return np.empty(0),[],[]
    cls_scores   = scores[inds, c]
    cls_boxes    = mean_boxes[inds, c * 4:(c + 1) * 4]
    #[cls_var,cls_boxes,cls_scores]
    cls_dets = np.hstack((cls_boxes.cpu().numpy(), cls_scores.unsqueeze(1).cpu().numpy())) \
        .astype(np.float32, copy=False)
    keep = nms(cls_boxes, cls_scores,
               cfg.TEST.NMS).cpu().numpy() if cls_dets.size > 0 else []
    cls_dets = cls_dets[keep, :]
    #Only if this variable has been provided
    return cls_dets, inds, keep



#TODO: Could use original imwidth/imheight
def filter_pred(rois, cls_score, a_cls_entropy, a_cls_var, e_cls_mutual_info, bbox_pred, a_bbox_var, e_bbox_var, imheight, imwidth, imscale, num_classes,thresh=0.1):
    #print('validation img properties h: {} w: {} s: {} '.format(imheight,imwidth,imscale))
    rois = rois[:, 1:5].detach().cpu().numpy()
    #Deleting extra dim
    #cls_score = np.reshape(cls_score, [cls_score.shape[0], -1])
    #bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])
    #torch.set_printoptions(profile="full")
    #print(bbox_pred)
    pred_boxes = bbox_pred
    
    #pred_boxes = bbox_transform_inv(torch.from_numpy(rois), torch.from_numpy(bbox_pred)).numpy()
    # x1 >= 0
    pred_boxes[:, 0::4] = torch.clamp_min(pred_boxes[:, 0::4],0)
    # y1 >= 0
    #pred_boxes[:, 1::4] = torch.max(pred_boxes[:, 1::4], 0,dim=1)[0]
    pred_boxes[:, 1::4] = torch.clamp_min(pred_boxes[:, 1::4], 0)
    # x2 < imwidth
    #pred_boxes[:, 2::4] = torch.min(pred_boxes[:, 2::4], torch.tensor([imwidth/imscale - 1]).cuda(),dim=1)[0]
    pred_boxes[:, 2::4] = torch.clamp_max(pred_boxes[:, 2::4],imwidth/imscale - 1)
    # y2 < imheight 3::4 means start at 3 then jump every 4
    #pred_boxes[:, 3::4] = torch.min(pred_boxes[:, 3::4], torch.tensor([imheight/imscale - 1]).cuda(),dim=1)[0]
    pred_boxes[:, 3::4] = torch.clamp_max(pred_boxes[:, 3::4],imheight/imscale - 1)
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
        all_box, inds, keep = nms_hstack_torch(cls_score,pred_boxes,thresh,j)
        uncertainties = {}
        if(len(keep) != 0):
            if(cfg.ENABLE_ALEATORIC_BBOX_VAR):
                uncertainties['a_bbox_var'] = nms_hstack_var_torch('bbox',a_bbox_var,inds,keep,j)
            else:
                a_bbox_var = [0,0,0,0]
            if(cfg.ENABLE_EPISTEMIC_BBOX_VAR):
                uncertainties['e_bbox_var'] = nms_hstack_var_torch('bbox',e_bbox_var,inds,keep,j)
            else:
                e_bbox_var = [0,0,0,0]
            if(cfg.ENABLE_ALEATORIC_CLS_VAR):
                #uncertainties['a_cls_entropy'] = a_cls_entropy
                uncertainties['a_cls_entropy'] = nms_hstack_var_torch('cls',a_cls_entropy,inds,keep,j)
                uncertainties['a_cls_var']     = nms_hstack_var_torch('cls_var',a_cls_var,inds,keep,j)
            else:
                a_cls_var = [0]
            if(cfg.ENABLE_EPISTEMIC_CLS_VAR):
                uncertainties['e_cls_mutual_info'] = nms_hstack_var_torch('cls',e_cls_mutual_info,inds,keep,j)
            else:
                e_cls_var = [0]
        all_uncertainty[j] = uncertainties
        all_boxes[j] = all_box
    return rois, all_boxes, all_uncertainty
