# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import xml.etree.ElementTree as ET
import os
from model.config import cfg, get_output_dir
from shapely.geometry import Polygon
import pickle
import numpy as np
import utils.bbox as bbox_utils
from scipy.interpolate import InterpolatedUnivariateSpline
import sys
import operator
import json
import re
from scipy.spatial import ConvexHull
import utils.eval_utils as eval_utils

#Values    Name      Description
#----------------------------------------------------------------------------
#   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
#                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
#                     'Misc' or 'DontCare'
#   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
#                     truncated refers to the object leaving frame boundaries
#   1    occluded     Integer (0,1,2,3) indicating occlusion state:
#                     0 = fully visible, 1 = partly occluded
#                     2 = largely occluded, 3 = unknown
#   1    alpha        Observation angle of object, ranging [-pi..pi]
#   4    bbox         2D bounding box of object in the frame (0-based index):
#                     contains left, top, right, bottom pixel coordinates
#   3    dimensions   3D object dimensions: height, width, length (in meters)
#   3    location     3D object location x,y,z in camera coordinates (in meters)
#   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
#   1    score        Only for results: Float, indicating confidence in
#                     detection, needed for p/r curves, higher is better.

def waymo_eval(detpath,
               db,
               frameset,
               classname,
               cachedir,
               mode,
               ovthresh=0.5,
               eval_type='2d',
               d_levels=0):
    #Min overlap is 0.7 for cars, 0.5 for ped/bike
    """rec, prec, ap = waymo_eval(detpath,
                              annopath,
                              framesetfile,
                              classname,
                              [ovthresh])

  Top level function that does the PASCAL VOC evaluation.

  detpath: Path to detections
      detpath.format(classname) should produce the detection results file.
  annopath: Path to annotations
      annopath.format(framename) should be the xml annotations file.
  framesetfile: Text file containing the list of frames, one frame per line.
  classname: Category name (duh)
  cachedir: Directory for caching the annotations
  [ovthresh]: Overlap threshold (default = 0.5)
  [use_07_metric]: Whether to use VOC07's 11 point AP computation
      (default False)
  """

    #Misc hardcoded variables
    idx = 0
    ovthresh_dc = 0.5
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(framename)
    # assumes framesetfile is a text file with each line an frame name
    # cachedir caches the annotations in a pickle file

    frame_path = eval_utils.get_frame_path(db, mode, eval_type)
    labels_filename = eval_utils.get_labels_filename(db, eval_type)

    class_recs = eval_utils.load_recs(frameset, frame_path, labels_filename, db, mode, classname)
    # read dets
    detfile = detpath.format(classname)
    print('Opening det file: ' + detfile)
    #sys.exit('donezo')
    with open(detfile, 'r') as f:
        lines = f.readlines()
    #Extract detection file into array
    splitlines   = [x.strip().split(' ') for x in lines]
    #Many entries have the same idx & token
    frame_idx    = [x[0] for x in splitlines] #TODO: I dont like how this is along many frames
    frame_tokens = [x[1] for x in splitlines]
    confidence   = np.array([float(x[2]) for x in splitlines])
    #All detections for specific class
    bbox_elem  = cfg[cfg.NET_TYPE.upper()].NUM_BBOX_ELEM
    BB         = np.array([[float(z) for z in x[3:3+bbox_elem]] for x in splitlines])
    det_cnt    = np.zeros((cfg.NUM_SCENES,cfg.MAX_IMG_PER_SCENE))
    scene_desc = ["" for x in range(cfg.NUM_SCENES)]
    uc_avg, uncertainties = eval_utils.extract_uncertainties(bbox_elem,splitlines)
    #Repeated for X detections along every frame presented
    idx = len(frame_idx)
    #DEPRECATED ---- 3 types, easy medium hard
    tp         = np.zeros((idx,d_levels))
    fp         = np.zeros((idx,d_levels))
    fn         = np.zeros((idx))
    tp_frame   = np.zeros(cfg.NUM_SCENES*cfg.MAX_IMG_PER_SCENE)
    fp_frame   = np.zeros(cfg.NUM_SCENES*cfg.MAX_IMG_PER_SCENE)
    npos_frame = np.zeros(cfg.NUM_SCENES*cfg.MAX_IMG_PER_SCENE)
    npos       = np.zeros((len(class_recs),d_levels))
    #Count number of total labels in all frames
    count_npos(class_recs, npos, npos_frame)
    det_results         = []
    scene_uncertainties = []
    #Check if there are any dets at all
    if BB.shape[0] > 0:
        # sort by confidence (highest first)
        sorted_ind    = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        idx_sorted          = [int(frame_idx[x]) for x in sorted_ind]
        frame_tokens_sorted = [frame_tokens[x] for x in sorted_ind]
        #print(frame_ids)

        # go down dets and mark true positives and false positives
        #Zip together sorted_ind with frame tokens sorted. 
        #sorted_ind -> Needed to know which detection we are selecting next
        #frame_tokens_sorted -> Needed to know which set of GT's are for the same frame as the det
        print('num dets {}'.format(len(sorted_ind)))
        for det_idx,token in zip(sorted_ind,frame_tokens_sorted):
            #R is a subset of detections for a specific class
            #print('doing det for frame {}'.format(frame_idx[d]))
            #Need to find associated GT frame ID alongside its detection id 'd'
            #Only one such frame, why appending?
            R = None
            skip_iter = True
            R = eval_utils.find_rec(class_recs,token)
            if(R is None):
                continue
            #Deprecated
            #R = class_recs[frame_ids[d]]
            bb = BB[det_idx, :].astype(float)
            var = {}
            #Variance extraction, collect on a per scene basis
            for key,val in uncertainties.items():
                uc_avg[key][R['scene_idx']] += val[det_idx, :]
                var[key] = val[det_idx, :]

            det_cnt[R['scene_idx']][R['frame_idx']] += 1
            scene_desc[R['scene_idx']] = R['scene_desc']
            ovmax = -np.inf
            #Multiple possible bounding boxes, perhaps for multi car detection
            BBGT = R['boxes'].astype(float)
            BBGT_dc = R['boxes_dc'].astype(float)
            #Preload all GT boxes and count number of true positive GT's
            #Not sure why we're setting ignore to false here if it were true
            #for i, BBGT_elem in enumerate(BBGT):
            #    BBGT_height = BBGT_elem[3] - BBGT_elem[1]
            ovmax_dc = 0
            if BBGT_dc.size > 0 and cfg.TEST.IGNORE_DC:
                overlaps_dc = eval_utils.iou(BBGT_dc,bb,eval_type)
                ovmax_dc = np.max(overlaps_dc)
            #Compute IoU
            if BBGT.size > 0:
                overlaps = eval_utils.iou(BBGT,bb,eval_type)
                ovmax = np.max(overlaps)
                #Index of max overlap between a BBGT and BB
                jmax = np.argmax(overlaps)
            else:
                jmax = 0
            # Minimum IoU Threshold for a true positive
            if ovmax > ovthresh and ovmax_dc < ovthresh_dc:
                #if ovmax > ovthresh:
                #ignore if not contained within easy, medium, hard
                if not R['ignore'][jmax]:
                    if not R['hit'][jmax]:
                        if(R['difficulty'][jmax] < 2):
                            tp[det_idx,0] += 1
                        tp[det_idx,1] += 1
                        tp_frame[int(R['idx'])] += 1
                        R['hit'][jmax] = True
                        det_results.append(eval_utils.write_det(R,bb,var,jmax))
                    else:
                        #If it already exists, cant double classify on same spot.
                        if(R['difficulty'][jmax] < 2):
                            fp[det_idx,0] += 1
                        fp[det_idx,1] += 1
                        fp_frame[int(R['idx'])] += 1
                        det_results.append(eval_utils.write_det(R,bb,var))
            #If your IoU is less than required, its simply a false positive.
            elif(BBGT.size > 0 and ovmax_dc < ovthresh_dc):
                #elif(BBGT.size > 0)
                fp[det_idx,0] += 1
                fp[det_idx,1] += 1
                fp_frame[int(R['idx'])] += 1
                det_results.append(eval_utils.write_det(R,bb,var))
    else:
        print('waymo eval, no GT boxes detected')
    for i in np.arange(cfg.NUM_SCENES):
        scene_dets = np.sum(det_cnt[i])
        scene_uc = eval_utils.write_scene_uncertainty(uc_avg,scene_dets,i)
        if(scene_uc != '' and cfg.DEBUG.PRINT_SCENE_RESULT):
            print(scene_uc)
            scene_uncertainties.append(scene_uc)

    if(cfg.DEBUG.TEST_FRAME_PRINT):
        eval_utils.display_frame_counts(tp_frame,fp_frame,npos_frame)
    out_dir = get_output_dir(db,mode='test')
    out_file = '{}_detection_results.txt'.format(classname)
    eval_utils.save_detection_results(det_results, out_dir, out_file)
    if(len(scene_uncertainties) != 0):
        scene_out_file = '{}_scene_uncertainty_results.txt'.format(classname)
        eval_utils.save_detection_results(scene_uncertainties, out_dir, scene_out_file)

    map = mrec = mprec = np.zeros((d_levels,))
    prec = 0
    rec  = 0
    fp_sum = np.cumsum(fp, axis=0)
    tp_sum = np.cumsum(tp, axis=0)
    #fn     = 1-fp
    #fn_sum = np.cumsum(fn, axis=0)
    npos_sum = np.sum(npos, axis=0)
    #print('Difficulty Level: {:d}, fp sum: {:f}, tp sum: {:f} npos: {:d}'.format(i, fp_sum[i], tp_sum[i], npos[i]))
    #recall
    #Per frame per class AP
    for i in range(0,d_levels):
        npos_sum_d = npos_sum[i]
        #Override to avoid NaN
        if(npos_sum_d == 0):
            npos_sum_d = np.sum([1])
        rec = tp_sum[:,i] / npos_sum_d.astype(float)
        prec = tp_sum[:,i] / np.maximum(tp_sum[:,i] + fp_sum[:,i], np.finfo(np.float64).eps)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth precision
        rec, prec = zip(*sorted(zip(rec, prec)))
        mprec[i]  = np.average(prec)
        mrec[i]   = np.average(rec)
        map[i]    = eval_utils.ap(rec, prec)
    return mrec, mprec, map


def count_npos(class_recs, npos, npos_frame):
    for i, rec in enumerate(class_recs):
        if(rec['ignore_frame'] is False):
            for j, ignore_elem in enumerate(rec['ignore']):
                if(not ignore_elem):
                    if(rec['difficulty'][j] < 2):
                        npos[i,0] += 1
                    npos[i,1] += 1
                    npos_frame[int(rec['idx'])] += 1