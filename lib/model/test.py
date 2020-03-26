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
try:
    import cPickle as pickle
except ImportError:
    import pickle
import os
import shutil
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils.timer import Timer
from torchvision.ops import nms
from utils.blob import im_list_to_blob

from model.config import cfg, get_output_dir
from model.bbox_transform import clip_boxes, bbox_transform_inv

import torch
from utils.filter_predictions import filter_and_draw_prep

def _get_image_blob(im):
    """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
    im_orig = im.astype(np.float32, copy=True)

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(
            im_orig,
            None,
            None,
            fx=im_scale,
            fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        im = im.astype(np.float32)
        im -= cfg.PIXEL_MEANS
        im = im/cfg.PIXEL_STDDEVS
        im = im[:,:,cfg.PIXEL_ARRANGE]
        processed_ims.append(im)
    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)


def _get_blobs(im):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {}
    blobs['data'], im_scale_factors = _get_image_blob(im)

    return blobs, im_scale_factors


def _clip_boxes(boxes, frame_shape):
    """Clip boxes to image boundaries."""
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(boxes[:, 0::4], frame_shape[0])
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(boxes[:, 1::4], frame_shape[2])
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.minimum(boxes[:, 2::4], frame_shape[1])
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.minimum(boxes[:, 3::4], frame_shape[3])
    return boxes


def _rescale_boxes(boxes, inds, scales):
    """Rescale boxes according to image rescaling."""
    for i in range(boxes.shape[0]):
        boxes[i, :] = boxes[i, :] / scales[int(inds[i])]

    return boxes


def im_detect(net, im, num_classes, thresh):
    blobs, im_scales = _get_blobs(im)
    assert len(im_scales) == 1, "Only single-image batch implemented"

    im_blob = blobs['data']
    blobs['info'] = np.array(
        [im_blob.shape[1], im_blob.shape[2], im_scales[0]], dtype=np.float32)
    net.set_num_mc_run(cfg.NUM_MC_RUNS)
    _, probs, bbox_pred, uncertainties, rois = net.test_frame(blobs['data'],blobs['info'])
    net.set_num_mc_run(1)
    boxes = rois[:, 1:5]
    #TODO: Useless??
    #probs = np.reshape(probs, [probs.shape[0], -1])
    #bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])
    #e^(x) where x = log(bbox_var) 


    rois, bbox_pred, uncertainties = filter_and_draw_prep(probs,
                                                          boxes
                                                          blobs['info'],
                                                          num_classes,thresh)
    #TODO: Fix all this bullshit
    #REPLACED BY - filter_pred()
    #if(cfg.ENABLE_ALEATORIC_BBOX_VAR):
    #    bbox_var = np.exp(bbox_var)
    #    num_sample = cfg.TEST.NUM_BBOX_VAR_SAMPLE
    #    #TODO: Fix magic /10.0
    #    sample = np.random.normal(size=(bbox_var.shape[0],bbox_var.shape[1],num_sample))*bbox_var[:,:,None]/10.0 + bbox_pred[:,:,None]
    #    sample = sample.astype(np.float32)
    #else:
    #    num_sample = 1
    #    sample = np.repeat(bbox_pred[:,:,np.newaxis],num_sample,axis=2)
    #boxes = np.repeat(boxes[:,:,np.newaxis],num_sample,axis=2)
    #pred_boxes = np.zeros(sample.shape)
    #if cfg.TEST.BBOX_REG:
    #    # Apply bounding-box regression deltas
    #    for i in range(num_sample):
    #        box_deltas = sample.permute(0,2,1).view(-1,sample.shape[1]).view(-1,num_sample,sample.shape[1])
    #        pred_boxes = bbox_transform_inv(boxes, box_deltas,im_scales[2]).numpy()
    #        pred_boxes = _clip_boxes(pred_boxes, im.shape)
    #else:
        # Simply repeat the boxes, once for each class
   #     pred_boxes = np.tile(boxes, (1, probs.shape[1], cfg.TEST.NUM_BBOX_VAR_SAMPLE))

    return rois, bbox_pred, uncertainties

def score_to_color_dict(score):
    color = 'black'
    if(score <= 0.40):
        color = 'white'
    elif(score <= 0.50):
        color = 'lavenderblush'
    elif(score <= 0.60):
        color = 'lightpink'
    elif(score <= 0.70):
        color = 'pink'
    elif(score <= 0.80):
        color = 'violet'
    elif(score <= 0.90):
        color = 'palevioletred'
    elif(score <= 0.95):
        color = 'crimson'
    else:
        color = 'red'
    return color

def test_net(net, imdb, out_dir, max_per_image=100, thresh=0.1, mode='test',draw_det=False,eval_det=False):
    np.random.seed(cfg.RNG_SEED)
    #TODO: Check if a cached detections exists
    """Test a Fast R-CNN network on an image database."""
    if(mode == 'test'):
        num_images = len(imdb._test_image_index)
    elif(mode == 'val'):
        num_images = len(imdb._val_image_index)
    # all detections are collected into:
    #  all_boxes[cls][image] = N x 5 array of detections in
    #  (x1, y1, x2, y2, score)
    num_uncertainty_en = 0
    num_uncertainty_pos = 0
    if(cfg.ENABLE_ALEATORIC_BBOX_VAR):
        num_uncertainty_en += 1
        num_uncertainty_pos += 4
    if(cfg.ENABLE_EPISTEMIC_BBOX_VAR):
        num_uncertainty_en += 1
        num_uncertainty_pos += 4
    if(cfg.ENABLE_ALEATORIC_CLS_VAR):
        num_uncertainty_en += 1
        num_uncertainty_pos += 1
    if(cfg.ENABLE_EPISTEMIC_CLS_VAR):
        num_uncertainty_en += 1
        num_uncertainty_pos += 1


    all_boxes       = [[[] for _ in range(num_images)]
                       for _ in range(imdb.num_classes)]
    if(num_uncertainty_en != 0):
        all_uncertainty = [[[[] for _ in range(num_uncertainty_en)] for _ in range(num_images)] for _ in range(imdb.num_classes)]
    output_dir = get_output_dir(imdb, out_dir)
    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}

    datapath = os.path.join(cfg.DATA_DIR, 'waymo',mode,'{}_drawn'.format(mode))
    print('deleting files in dir {}'.format(datapath))
    shutil.rmtree(datapath)
    os.makedirs(datapath)
    #for i in range(10):
    for i in range(num_images):
        #Start here, this is where we will grab scene info for each detection.
        #Variance comes up here, and is actually computed below.
        imfile = imdb.image_path_at(i,mode)
        im = cv2.imread(imfile)

        _t['im_detect'].tic()
        rois, bbox, uncertainties = im_detect(net, im, imdb.num_classes,thresh)
        _t['im_detect'].toc()
        _t['misc'].tic()
        if max_per_image > 0 and len(bbox[0]) != 0:
            image_scores = np.hstack(
                [bbox[j][:, -1] for j in range(1, imdb.num_classes)])
            #Grab highest X scores to keep for image
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, imdb.num_classes):
                    #-1  is last element in array, keep all rows with high enough scores
                    keep = np.where(bbox[j][:, -1] >= image_thresh)[0]
                    bbox[j] = bbox[j][keep, :]
                    for key in uncertainties[j]:
                        uncertainties[j][key] = uncertainties[j][key][keep, :]
        for j in range(1, imdb.num_classes):
            if(bbox[j].size != 0):
                bbox_uncertainty_hstack = np.zeros((bbox[j].shape[0],bbox[j].shape[1]+num_uncertainty_pos))
                bbox_uncertainty_hstack[:,0:bbox[j].shape[1]] = bbox[j]
                hstack_ptr = bbox[j].shape[1]
                for k,key in enumerate(uncertainties[j]):
                    uncert = np.array(uncertainties[j][key])
                    hstack_end = hstack_ptr+uncert.shape[1]
                    bbox_uncertainty_hstack[:,hstack_ptr:hstack_end] = uncert[:,:]
                    hstack_ptr = hstack_end
                all_boxes[j][i] = bbox_uncertainty_hstack
            else:
                all_boxes[j][i] = np.empty(0)
        _t['misc'].toc()

        print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s'
            .format(i + 1, num_images, _t['im_detect'].average_time(), _t['misc'].average_time()))
        
        #box is x1,y1,x2,y2 where x1,y1 is top left, x2,y2 is bottom right
        if(draw_det):
            image_boxes = []
            gt_boxes = imdb.find_gt_for_img(imfile,'val')
            if(gt_boxes is None):
                #print('Draw and save: image {} had no GT boxes'.format(imfile))
                imdb.draw_and_save_eval(imfile,[],[],bbox,uncertainties,0,mode)
            else:    
                imdb.draw_and_save_eval(imfile,gt_boxes['boxes'],gt_boxes['gt_classes'],bbox,uncertainties,0,mode)

    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
    if(eval_det):
        print('Evaluating detections')
        imdb.evaluate_detections(all_boxes, output_dir,'val')
