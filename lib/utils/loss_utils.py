from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

from model.config import cfg

#def huber_loss(pred, targets, huber_delta, sigma, sin_en=False):
#    sigma_2 = sigma**2
#    box_diff = pred - targets
#    if(sin_en):
#        box_diff = torch.sin(box_diff)
#    abs_in_box_diff = torch.abs(box_diff)
#    smoothL1_sign = (abs_in_box_diff < huber_delta / sigma_2).detach().float()
#    above_one = (abs_in_box_diff - (0.5 * huber_delta / sigma_2)) * (1. - smoothL1_sign)
#    below_one = torch.pow(box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign
#    in_loss_box = below_one + above_one
#    return in_loss_box

def huber_loss(pred, targets, huber_delta, sin_en=False):
    box_diff = pred - targets
    if(sin_en):
        box_diff = torch.sin(box_diff)
    abs_in_box_diff = torch.abs(box_diff)
    smoothL1_sign = (abs_in_box_diff < huber_delta).detach().float()
    above_one = huber_delta * (abs_in_box_diff - (0.5 * huber_delta)) * (1. - smoothL1_sign)
    below_one = 0.5 * torch.pow(box_diff, 2) * smoothL1_sign
    in_loss_box = below_one + above_one
    return in_loss_box

def smooth_l1_loss(stage,
                   bbox_pred,
                   bbox_targets,
                   bbox_var,
                   bbox_inside_weights,
                   bbox_outside_weights,
                   sigma=1.0,
                   dim=[1]):
    if((stage == 'RPN' and cfg.UC.EN_RPN_BBOX_ALEATORIC) or (stage == 'DET' and cfg.UC.EN_BBOX_ALEATORIC)):
        bbox_var_en = True
    else:
        bbox_var_en = False
    #Ignore diff when target is not a foreground target
    # a mask array for the foreground anchors (called “bbox_inside_weights”) is used to calculate the loss as a vector operation and avoid for-if loops.
    bbox_pred = bbox_pred*bbox_inside_weights
    bbox_targets = bbox_targets*bbox_inside_weights
    #torch.set_printoptions(profile="full")
    #print('from _smooth_l1_loss')
    #print(bbox_targets)
    #print(bbox_inside_weights)
    #torch.set_printoptions(profile="default")
    bias = 0
    if(cfg.NET_TYPE == 'lidar' and stage == 'DET'):
        bbox_shape   = [bbox_pred.shape[0],bbox_pred.shape[1]]
        elem_rm      = int(bbox_shape[1]/7)
        bbox_pred_aa = bbox_pred.reshape(-1,7)[:,0:6].reshape(-1,bbox_shape[1]-elem_rm)
        targets_aa   = bbox_targets.reshape(-1,7)[:,0:6].reshape(-1,bbox_shape[1]-elem_rm)
        #TODO: Sum across elements
        loss_box     = huber_loss(bbox_pred_aa,targets_aa,1.0)
        #TODO: Do i need to compute the sin of the difference here?
        sin_pred     = bbox_pred.reshape(-1,7)[:,6:7].reshape(-1,elem_rm)
        #Convert to sin to normalize, targets will be in degrees off of anchor
        sin_targets  = bbox_targets.reshape(-1,7)[:,6:7].reshape(-1,elem_rm)
        ry_loss      = huber_loss(sin_pred,sin_targets,1.0,sin_en=cfg.LIDAR.EN_RY_SIN)
        #self._losses['ry_loss'] = torch.mean(torch.sum(ry_loss,dim=1))
        in_loss_box  = torch.cat((loss_box.reshape(-1,6),ry_loss.reshape(-1,1)),dim=1).reshape(-1,bbox_shape[1])
        reg_loss_weight_tensor = torch.FloatTensor(cfg.LIDAR.REG_LOSS_WEIGHT).to(device=in_loss_box.device)
        in_loss_box_weighted  = in_loss_box.reshape(-1,7)*reg_loss_weight_tensor
        in_loss_box = in_loss_box_weighted.reshape(-1,in_loss_box.shape[1])
        #bbox_outside_weights = torch.mean(bbox_outside_weights,axis=1)
    else:
        in_loss_box = huber_loss(bbox_pred,bbox_targets,1.0)

    if(bbox_var_en):
        #Don't need covariance matrix as it collapses itself in the end anyway
        in_loss_box = 0.5*torch.exp(-bbox_var) + 0.5*bbox_var
        in_loss_box = in_loss_box*bbox_inside_weights
        #bias        = 1
        #torch.set_printoptions(profile="full")
        #print(in_loss_box[in_loss_box.nonzero()])
        #torch.set_printoptions(profile="default")
    #Used to normalize the predictions, this is only used in the RPN
    #By default negative(background) and positive(foreground) samples have equal weighting
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = out_loss_box
    #Condense down to 1D array, each entry is the box_loss for an individual box, array is batch size of all predicted boxes
    #[loss,y,x,num_anchor]
    for i in sorted(dim, reverse=True):
        loss_box = loss_box.sum(i)
    #print(loss_box.size())
    #TODO: Could it be mean is taken at a different level between rpn and 2nd stage??
    loss_box = loss_box.mean()
    return loss_box + bias

def compute_bbox_cov(bbox_samples):
    mc_bbox_mean = torch.mean(bbox_samples,dim=0)
    mc_bbox_pred = bbox_samples.unsqueeze(3)
    mc_bbox_var = torch.mean(torch.matmul(mc_bbox_pred,mc_bbox_pred.transpose(2,3)),dim=0)
    mc_bbox_mean = mc_bbox_mean.unsqueeze(2)
    mc_bbox_var = mc_bbox_var - torch.matmul(mc_bbox_mean,mc_bbox_mean.transpose(1,2))
    mc_bbox_var = mc_bbox_var*torch.eye(mc_bbox_var.shape[-1]).cuda()
    mc_bbox_var = torch.sum(mc_bbox_var,dim=-1)
    #mc_bbox_var = torch.diag_embed(mc_bbox_var,offset=0,dim1=1,dim2=2)
    return mc_bbox_var.clamp_min(0.0)

def compute_bbox_var(bbox_samples):
    n = bbox_samples.shape[0]
    mc_bbox_mean = torch.pow(torch.sum(bbox_samples,dim=0),2)
    mc_bbox_var = torch.sum(torch.pow(bbox_samples,2),dim=0)
    mc_bbox_var += -mc_bbox_mean/n
    mc_bbox_var = mc_bbox_var/(n-1)
    return mc_bbox_var.clamp_min(0.0)

def categorical_entropy(cls_prob):
    #Compute entropy for each class(y=c)
    cls_entropy = cls_prob*torch.log2(cls_prob)
    #Sum across classes
    total_entropy = -torch.sum(cls_entropy,dim=1)
    #true_cls      = torch.gather(cls_score,1,labels.unsqueeze(1)).squeeze(1)
    #softmax = torch.exp(true_cls)/torch.mean(torch.exp(cls_score),dim=1)
    return total_entropy
#input: cls_score (T,N,C) T-> Samples, N->Batch, C-> Classes

def categorical_mutual_information(cls_score):
    cls_prob = F.softmax(cls_score,dim=2)
    avg_cls_prob = torch.mean(cls_prob,dim=0)
    total_entropy = categorical_entropy(avg_cls_prob)
    #Take sum of entropy across classes
    mutual_info = torch.sum(cls_prob*torch.log2(cls_prob),dim=2)
    #Get expectation over T forward passes
    mutual_info = torch.mean(mutual_info,dim=0)
    mutual_info += total_entropy
    return mutual_info

def logit_distort(cls_score, cls_var, num_sample):
    distribution     = torch.distributions.Normal(0,torch.sqrt(cls_var))
    cls_score_resize = cls_score.repeat(num_sample,1,1)
    logit_samples    = distribution.sample((num_sample,)) + cls_score_resize
    return logit_samples

def bayesian_cross_entropy(cls_score,cls_var,targets,num_sample):
    #Step 1: Create target mask to shift 'correct' logits
    cls_score_mask = torch.zeros_like(cls_score).scatter_(1, targets.unsqueeze(1), 1)
    cls_score_shifted = cls_score  #- cls_score_mask
    #Step 2: Get a set of distorted logits sampled from a gaussian distribution
    logit_samples    = logit_distort(cls_score_shifted,cls_var,num_sample)
    #Step 3: Perform softmax over all distorted logits
    softmax_samples  = F.softmax(logit_samples,dim=2)
    #Step 4: Take average of T distorted samples
    avg_softmax     = torch.mean(softmax_samples,dim=0)
    #Step 5: Perform negative log likelihood
    log_avg_softmax = torch.log(avg_softmax)
    sel_log_conf    = -log_avg_softmax.gather(1,targets.unsqueeze(1))
    #ce_loss         = F.nll_loss(log_avg_softmax,targets)
    #Step 6: Add regularizer
    #ce_loss         += 0.05*torch.mean(cls_var,dim=1)
    #Step 7: Take average over proposals
    ce_loss          = torch.mean(sel_log_conf)
    #Compute mutual info from logit samples for display on tensorboard
    a_mutual_info   = categorical_mutual_information(logit_samples)
    return ce_loss, a_mutual_info

def bayesian_cross_entropy_simple(cls_score,cls_var,targets,num_sample):
    #Step 1: Create target mask to shift 'correct' logits
    cls_score_mask = torch.zeros_like(cls_score).scatter_(1, targets.unsqueeze(1), 1)
    cls_score_shifted = cls_score  #- cls_score_mask
    #Step 2: Get a set of distorted logits sampled from a gaussian distribution
    logit_samples    = logit_distort(cls_score_shifted,cls_var,num_sample)
    #Step 4: Match targets to same size as logit samples
    target_samples = targets.unsqueeze(1).repeat(1,num_sample)
    logit_samples  = logit_samples.permute((1,2,0))
    #Step 3: Perform cross entropy loss
    ce_loss  = F.cross_entropy(logit_samples,target_samples)
    #Compute mutual info from logit samples for display on tensorboard
    a_mutual_info   = categorical_mutual_information(logit_samples)
    return ce_loss, a_mutual_info


def bayesian_elu_cross_entropy(cls_score,cls_var,targets,num_sample):
    #Step 1: Get undistorted cross entropy loss
    undist_ce_loss   = F.cross_entropy(cls_score,targets,reduction='none')
    #Step 2: Get a set of distorted logits sampled from a gaussian distribution
    logit_samples    = logit_distort(cls_score,cls_var,num_sample)
    #Step 3: Perform softmax over all distorted logits
    softmax_samples  = F.softmax(logit_samples,dim=2)
    #Step 4: Take average of T distorted samples
    avg_softmax     = torch.mean(softmax_samples,dim=0)
    #Step 5: Perform negative log likelihood
    dist_ce_loss    = F.nll_loss(torch.log(avg_softmax),targets,reduction='none')
    #Step 6: Get difference between distorted and undistorted loss
    diff = undist_ce_loss - dist_ce_loss
    #Step 7: Apply elu
    diff_elu = -F.elu(-diff)
    #Step 8: Form regularizer
    regularizer = torch.mean(cls_var,dim=1)  #Average across both logits
    #Step 8: Combine terms and add regularizer
    ce_loss         = diff_elu*undist_ce_loss + undist_ce_loss + torch.exp(regularizer) - torch.ones_like(regularizer)
    ce_loss         = torch.mean(ce_loss)
    #Compute mutual info from logit samples for display on tensorboard
    a_mutual_info   = categorical_mutual_information(logit_samples)
    return ce_loss, a_mutual_info