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
from roi_data_layer import minibatch
import torch
from utils.filter_predictions import filter_and_draw_prep
import utils.bbox as bbox_utils

def _get_blobs(filename):
    """Convert a frame and RoIs within that image into network inputs."""
    blobs = {}
    if(cfg.NET_TYPE == 'image'):
        scale = cfg.TEST.SCALES[0]
        infos, blobs['data'], _ = minibatch._get_image_blob(filename,scale,augment_en=cfg.TEST.AUGMENT_EN,mode='test')
        blobs['info'] = infos[0]
    elif(cfg.NET_TYPE == 'lidar'):
        scale = cfg.TEST.SCALES[0]
        area_extents = [cfg.LIDAR.X_RANGE[0],cfg.LIDAR.Y_RANGE[0],cfg.LIDAR.Z_RANGE[0],cfg.LIDAR.X_RANGE[1],cfg.LIDAR.Y_RANGE[1],cfg.LIDAR.Z_RANGE[1]]
        infos, blobs['data'], _ = minibatch._get_lidar_blob(filename,area_extents,scale,augment_en=False,mode='test')
        blobs['info'] = infos[0]
    return blobs


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


def frame_detect(net, blobs, num_classes, thresh):
    blob = blobs['data']
    #DEPRECATED
    #blobs['info'] = np.array(
    #    [blob.shape[1], blob.shape[2], scales[0]], dtype=np.float32)
    if(cfg.UC.EN_BBOX_EPISTEMIC or cfg.UC.EN_CLS_EPISTEMIC):
        net.set_e_num_sample(cfg.UC.E_NUM_SAMPLE)
    _, probs, bbox_pred, rois, uncertainties = net.test_frame(blobs['data'],blobs['info'])
    if(cfg.UC.EN_BBOX_EPISTEMIC or cfg.UC.EN_CLS_EPISTEMIC):
        net.set_e_num_sample(1)
    #boxes = rois[:, 1:5]
    boxes = bbox_pred
    #TODO: Useless??
    #probs = np.reshape(probs, [probs.shape[0], -1])
    #bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])
    #e^(x) where x = log(bbox_var) 


    rois, bbox_pred, uncertainties = filter_and_draw_prep(rois,
                                                          probs,
                                                          boxes,
                                                          uncertainties,
                                                          blobs['info'],
                                                          num_classes,
                                                          thresh,
                                                          cfg.NET_TYPE)
    #TODO: Fix all this bullshit
    #REPLACED BY - filter_pred()
    #if(cfg.UC.EN_BBOX_ALEATORIC):
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

def test_net(net, db, out_dir, max_dets=100, thresh=0.1, mode='test',draw_det=False,eval_det=False):
    np.random.seed(cfg.RNG_SEED)
    #TODO: Check if a cached detections exists
    """Test a Fast R-CNN network on an image database."""
    if(mode == 'test'):
        num_images = len(db._test_index)
    elif(mode == 'val'):
        num_images = len(db._val_index)
    else:
        num_images = 0
    # all detections are collected into:
    #  all_boxes[cls][image] = N x 5 array of detections in
    #  (x1, y1, x2, y2, score)
    num_uncertainty_pos = 0
    if(cfg.UC.EN_BBOX_ALEATORIC):
        num_uncertainty_pos += cfg[cfg.NET_TYPE.upper()].NUM_BBOX_ELEM
    if(cfg.UC.EN_BBOX_EPISTEMIC):
        num_uncertainty_pos += cfg[cfg.NET_TYPE.upper()].NUM_BBOX_ELEM
    if(cfg.UC.EN_CLS_ALEATORIC):
        num_uncertainty_pos += 4
    if(cfg.UC.EN_CLS_EPISTEMIC):
        num_uncertainty_pos += 4


    all_boxes       = [[[] for _ in range(num_images)]
                       for _ in range(db.num_classes)]
    #TODO: Output dir might need to be a bit more specific to run parallel experiments
    output_dir = get_output_dir(db, mode='test')
    if(os.path.isdir(output_dir)):
        print('deleting old test dir')
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    # timers
    _t = {'preload': Timer(), 'frame_detect': Timer(), 'misc': Timer()}
    print('testing on {} frames'.format(len(db._val_index)))

    #TODO: Move to the db
    test_mode = 'test'
    db.delete_eval_draw_folder(test_mode,test_mode)
    #datapath = os.path.join(cfg.DATA_DIR, 'waymo',mode,'{}_drawn'.format(mode))
    #print('deleting files in dir {}'.format(datapath))
    #shutil.rmtree(datapath)
    #os.makedirs(datapath)
    area_extents = [cfg.LIDAR.X_RANGE[0],cfg.LIDAR.Y_RANGE[0],cfg.LIDAR.Z_RANGE[0],cfg.LIDAR.X_RANGE[1],cfg.LIDAR.Y_RANGE[1],cfg.LIDAR.Z_RANGE[1]]
    #for i in range(10):
    for i in range(num_images):
        #Start here, this is where we will grab scene info for each detection.
        #Variance comes up here, and is actually computed below.
        if(cfg.NET_TYPE == 'image'):
            filename = db.path_at(i,mode)
            #print(filename)
            #frame = cv2.imread(filename)
            #np.set_printoptions(threshold=np.inf)
        elif(cfg.NET_TYPE == 'lidar'):
            filename = db.path_at(i,mode)
            #frame = db._load_pc(filename)
        #Put frame into array, single element only supported
        #frame = [frame]
        _t['preload'].tic()
        blobs = _get_blobs([filename])
        _t['preload'].toc()
        #assert len(blobs) == 1, "Only single-image batch implemented"
        _t['frame_detect'].tic()
        rois, bbox, uncertainties = frame_detect(net, blobs, db.num_classes,thresh)
        _t['frame_detect'].toc()
        _t['misc'].tic()
        #Stack output file with uncertainties
        for j in range(1, db.num_classes):
            cls_uncertainties = uncertainties[j]
            cls_boxes         = bbox[j].copy()
            if max_dets > 0 and len(cls_boxes) != 0:
                scores = cls_boxes[:, -1]
                #If we have too many detections, remove any past max allowed with a low score
                if len(scores) > max_dets:
                    filter_thresh = np.sort(scores)[-max_dets]
                    keep = np.where(cls_boxes[:, -1] >= filter_thresh)[0]
                    cls_boxes = cls_boxes[keep, :]
                    for key in cls_uncertainties:
                        cls_uncertainties[key] = cls_uncertainties[key][keep, :]
            if(cls_boxes.size != 0):
                if(cfg.NET_TYPE == 'lidar'):
                    cls_boxes = bbox_utils.bbox_voxel_grid_to_pc(cls_boxes,area_extents,blobs['info'])
                bbox_uncertainty_hstack = stack_uncertainties(cls_boxes,cls_uncertainties,num_uncertainty_pos)
                all_boxes[j][i] = bbox_uncertainty_hstack
            else:
                all_boxes[j][i] = np.empty(0)
        
        #box is x1,y1,x2,y2 where x1,y1 is top left, x2,y2 is bottom right
        if(draw_det):
            image_boxes = []
            gt_boxes = db.find_gt_for_frame(filename,'val')
            if(gt_boxes is None):
                print('Draw and save: image {} had no GT boxes'.format(filename))
                db.draw_and_save_eval(filename,[],[],bbox,uncertainties,0,test_mode)
            else:   
                if(cfg.NET_TYPE == 'lidar'):
                    boxes = bbox_utils.bbox_pc_to_voxel_grid(gt_boxes['boxes'],area_extents,blobs['info'])
                elif(cfg.NET_TYPE == 'image'):
                    boxes = gt_boxes['boxes']
                db.draw_and_save_eval(filename,boxes,gt_boxes['gt_classes'],bbox,uncertainties,0,test_mode,frame_arr=blobs['data'])

        _t['misc'].toc()
        if(cfg.DEBUG.EN_TEST_MSG):
            print('frame_detect: {:d}/{:d} preproc: {:4.3f}s | frame_det: {:4.3f}s | Misc&Draw: {:4.3f}s'.format(i + 1,
                                                                                                                num_images,
                                                                                                                _t['preload'].average_time(),
                                                                                                                _t['frame_detect'].average_time(),
                                                                                                                _t['misc'].average_time()))

    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
    if(eval_det):
        print('Evaluating detections')
        db.evaluate_detections(all_boxes, output_dir,'val')


def stack_uncertainties(cls_bbox,cls_uncertainties, num_uc_pos):
    #Stack output file with uncertainties
    bbox_uncertainty_hstack = np.zeros((cls_bbox.shape[0],cls_bbox.shape[1]+num_uc_pos))
    bbox_uncertainty_hstack[:,0:cls_bbox.shape[1]] = cls_bbox
    hstack_ptr = cls_bbox.shape[1]
    for key, val in cls_uncertainties.items():
        uncert = val
        hstack_end = hstack_ptr+uncert.shape[1]
        bbox_uncertainty_hstack[:,hstack_ptr:hstack_end] = uncert[:,:]
        hstack_ptr = hstack_end
    return bbox_uncertainty_hstack
