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
from model.config import cfg
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


def parse_rec(filename):
    """ Parse the labels for the specific frame """
    label_lines = open(filename, 'r').readlines()
    objects = []
    for line in label_lines:
        label_arr = line.split(' ')
        obj_struct = {}
        obj_struct['name'] = label_arr[0]
        obj_struct['truncated'] = label_arr[1]
        obj_struct['occluded'] = label_arr[2]
        obj_struct['alpha'] = label_arr[3]
        obj_struct['bbox'] = [
            float(label_arr[4]),
            float(label_arr[5]),
            float(label_arr[6]),
            float(label_arr[7])
        ]
        obj_struct['3D_dim'] = [
            float(label_arr[8]),
            float(label_arr[9]),
            float(label_arr[10])
        ]
        obj_struct['3D_loc'] = [
            float(label_arr[11]),
            float(label_arr[12]),
            float(label_arr[13])
        ]
        obj_struct['rot_y'] = float(label_arr[14])
        objects.append(obj_struct)

    return objects


def waymo_ap(rec, prec):
    """ ap = voc_ap(rec, prec, [use_07_metric])
  Compute VOC AP given precision and recall.
  If use_07_metric is true, uses the
  VOC 07 11 point method (default:False).
  """
    #if use_07_metric:
        # 11 point metric
    #    ap = 0.
    #    for t in np.arange(0., 1.1, 0.1):
    #        if np.sum(rec >= t) == 0:
    #            p = 0
    #        else:
    #            p = np.max(prec[rec >= t])
    #        ap = ap + p / 11.
    #else:
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope (going backwards, precision will always increase as sorted by -confidence)
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def waymo_eval(detpath,
               db,
               frameset,
               classname,
               cachedir,
               mode,
               ovthresh=0.5,
               eval_type='2d'):
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
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(framename)
    # assumes framesetfile is a text file with each line an frame name
    # cachedir caches the annotations in a pickle file
    idx = 0
    # first load gt
    #if not os.path.isdir(cachedir):
    #    os.mkdir(cachedir)
    #cachefile = os.path.join(cachedir, '{}_{}_annots.pkl'.format(mode,classname))
    #if not os.path.isfile(cachefile):
        # load annotations
    if(eval_type == 'bev' or eval_type == '3d' or eval_type == 'bev_aa'):
        frame_path = os.path.join(db._devkit_path, mode, 'point_clouds')
        labels_filename = 'lidar_labels.json'
    elif(eval_type == '2d'):
        frame_path = os.path.join(db._devkit_path, mode, 'images')
        labels_filename = 'image_labels.json'

    class_recs = eval_load_recs(frameset, frame_path, labels_filename, db, mode, classname)
        # save
        #print('Saving cached annotations to {:s}'.format(cachefile))
        #with open(cachefile, 'wb') as f:
        #    pickle.dump(class_recs, f)
    #else:
        # load
    #    print('loading cached annotations from {:s}'.format(cachefile))
    #    with open(cachefile, 'rb') as f:
    #        try:
    #            class_recs = pickle.load(f)
    #        except:
    #            class_recs = pickle.load(f, encoding='bytes')

    #----------------------------------------------------
    ovthresh_dc = 0.5
    # read dets
    detfile = detpath.format(classname)
    print('Opening det file: ' + detfile)
    #sys.exit('donezo')
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    #Many entries have the same idx & token
    frame_idx = [x[0] for x in splitlines] #TODO: I dont like how this is along many frames
    frame_tokens = [x[1] for x in splitlines]
    confidence = np.array([float(x[2]) for x in splitlines])
    #All detections for specific class
    if(eval_type == '3d' or eval_type == 'bev' or eval_type == 'bev_aa'):
        BB = np.array([[float(z) for z in x[3:10]] for x in splitlines])
        #+3 due to 3 pieces of info before bounding box coords
        u_start = cfg.LIDAR.NUM_BBOX_ELEM + 3
    elif(eval_type == '2d'):
        BB = np.array([[float(z) for z in x[3:7]] for x in splitlines])
        u_start = cfg.IMAGE.NUM_BBOX_ELEM + 3
    else:
        print('invalid evaluation type {}'.format(eval_type))
        return
    #TODO: Add variance read here
    bbox_elem      = cfg[cfg.NET_TYPE.upper()].NUM_BBOX_ELEM
    uncertainties  = {}
    uc_avg         = {}
    det_cnt        = np.zeros((cfg.NUM_SCENES,cfg.MAX_IMG_PER_SCENE))
    scene_desc     = ["" for x in range(cfg.NUM_SCENES)]
    if(cfg.UC.EN_CLS_ALEATORIC):
        uc_avg['a_entropy']  = np.zeros((cfg.NUM_SCENES,cfg.MAX_IMG_PER_SCENE))
        uc_avg['a_mutual_info'] = np.zeros((cfg.NUM_SCENES,cfg.MAX_IMG_PER_SCENE))
        uc_avg['a_cls_var'] = np.zeros((cfg.NUM_SCENES,cfg.MAX_IMG_PER_SCENE))
        uncertainties['a_cls_var'] = np.array([[float(z) for z in x[u_start:u_start+1]] for x in splitlines])
        u_start += 1
        uncertainties['a_entropy'] = np.array([[float(z) for z in x[u_start:u_start+1]] for x in splitlines])
        u_start += 1
        uncertainties['a_mutual_info'] = np.array([[float(z) for z in x[u_start:u_start+1]] for x in splitlines])
        u_start += 1
        a_cls_var = np.array([[float(z) for z in x[u_start:u_start+1]] for x in splitlines])
        u_start += 1
    if(cfg.UC.EN_CLS_EPISTEMIC):
        uc_avg['e_entropy'] = np.zeros((cfg.NUM_SCENES,cfg.MAX_IMG_PER_SCENE))
        uc_avg['e_mutual_info'] = np.zeros((cfg.NUM_SCENES,cfg.MAX_IMG_PER_SCENE))
        uncertainties['e_entropy'] = np.array([[float(z) for z in x[u_start:u_start+1]] for x in splitlines])
        u_start += 1
        uncertainties['e_mutual_info'] = np.array([[float(z) for z in x[u_start:u_start+1]] for x in splitlines])
        u_start += 1
    if(cfg.UC.EN_BBOX_ALEATORIC):
        uc_avg['a_bbox_var'] = np.zeros((cfg.NUM_SCENES,bbox_elem))
        uncertainties['a_bbox_var'] = np.array([[float(z) for z in x[u_start:u_start+bbox_elem]] for x in splitlines])
        u_start += bbox_elem
    if(cfg.UC.EN_BBOX_EPISTEMIC):
        uc_avg['e_bbox_var'] = np.zeros((cfg.NUM_SCENES,bbox_elem))
        uncertainties['e_bbox_var'] = np.array([[float(z) for z in x[u_start:u_start+bbox_elem]] for x in splitlines])
        u_start += cfg.IMAGE.NUM_BBOX_ELEM
    #Repeated for X detections along every frame presented
    idx = len(frame_idx)
    #DEPRECATED ---- 3 types, easy medium hard
    tp = np.zeros((idx))
    fp = np.zeros((idx))
    fn = np.zeros((idx))
    tp_frame = np.zeros(cfg.NUM_SCENES*cfg.MAX_IMG_PER_SCENE)
    fp_frame = np.zeros(cfg.NUM_SCENES*cfg.MAX_IMG_PER_SCENE)
    npos_frame = np.zeros(cfg.NUM_SCENES*cfg.MAX_IMG_PER_SCENE)
    npos     = np.zeros(len(class_recs))
    #Count number of total labels in all frames
    for i, rec in enumerate(class_recs):
        if(rec['ignore_frame'] is False):
            for ignore_elem in rec['ignore']:
                if(not ignore_elem):
                    npos[i] += 1
                    npos_frame[int(rec['idx'])] += 1
    if BB.shape[0] > 0:
        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        #No need to sort BB if you just access via sorted_ind
        #BB = BB[sorted_ind, :]
        idx_sorted = [int(frame_idx[x]) for x in sorted_ind]
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
            for rec in class_recs:
                if(rec['idx'] == re.sub('[^0-9]','',token)):
                    if(rec['ignore_frame'] is False):
                        R = rec
                        skip_iter = False
                        break
            if(skip_iter):
                continue
            #Deprecated
            #R = class_recs[frame_ids[d]]
            bb = BB[det_idx, :].astype(float)
            #Variance extraction, collect on a per scene basis
            for key,var in uncertainties.items():
                uc_avg[key][R['scene_idx']] += var[det_idx, :]

            det_cnt[R['scene_idx']][R['frame_idx']] += 1
            scene_desc[R['scene_idx']] = R['scene_desc']
            ovmax = -np.inf
            cat = []
            #Multiple possible bounding boxes, perhaps for multi car detection
            BBGT = R['boxes'].astype(float)
            BBGT_dc = R['boxes_dc'].astype(float)
            jmax = 0
            #Preload all GT boxes and count number of true positive GT's
            #Not sure why we're setting ignore to false here if it were true
            #for i, BBGT_elem in enumerate(BBGT):
            #    BBGT_height = BBGT_elem[3] - BBGT_elem[1]
            ovmax_dc = 0
            if BBGT_dc.size > 0 and cfg.TEST.IGNORE_DC:
                overlaps_dc = eval_iou(BBGT_dc,bb,eval_type)
                ovmax_dc = np.max(overlaps_dc)
            #Compute IoU
            if BBGT.size > 0:
                overlaps = eval_iou(BBGT,bb,eval_type)
                ovmax = np.max(overlaps)
                #Index of max overlap between a BBGT and BB
                jmax = np.argmax(overlaps)
            # Minimum IoU Threshold for a true positive
            if ovmax > ovthresh and ovmax_dc < ovthresh_dc:
                #if ovmax > ovthresh:
                #ignore if not contained within easy, medium, hard
                if not R['ignore'][jmax]:
                    if not R['hit'][jmax]:
                        tp[det_idx] += 1
                        tp_frame[int(R['idx'])] += 1
                        R['hit'][jmax] = True
                    else:
                        #If it already exists, cant double classify on same spot.
                        fp[det_idx] += 1
                        fp_frame[int(R['idx'])] += 1
            #If your IoU is less than required, its simply a false positive.
            elif(BBGT.size > 0 and ovmax_dc < ovthresh_dc):
                #elif(BBGT.size > 0)
                fp[det_idx] += 1
                fp_frame[int(R['idx'])] += 1
    else:
        print('waymo eval, no GT boxes detected')


        for i in cfg.NUM_SCENES:
            scene_dets = np.sum(det_cnt[i])
            print_str = ''
            print_start = 'Scene: {} \n num_dets: {}'.format(i,scene_dets)
            if(scene_dets == 0):
                continue
            if(cfg.UC.EN_CLS_ALEATORIC):
                print_str += ' a_entropy,a_mutual_info,a_cls_var: '
                print_str += uc_avg['a_entropy'][i]/scene_dets
                print_str += ','
                print_str += uc_avg['a_mutual_info'][i]/scene_dets
                print_str += ','
                print_str += uc_avg['a_cls_var'][i]/scene_dets
            if(cfg.UC.EN_CLS_EPISTEMIC):
                print_str += ' e_entropy,e_mutual_info: '
                print_str += uc_avg['e_entropy'][i]/scene_dets
                print_str += ','
                print_str += uc_avg['e_mutual_info'][i]/scene_dets
            if(cfg.UC.EN_BBOX_ALEATORIC):
                print_str += ' a_bbox: '
                print_str += uc_avg['a_bbox_var'][i]/scene_dets
            if(cfg.UC.EN_BBOX_EPISTEMIC):
                print_str += ' e_bbox: '
                print_str += uc_avg['e_bbox_var'][i]/scene_dets
            if(print_str != ''):
                print(print_start)
                print(print_str)
    
    if(cfg.DEBUG.TEST_FRAME_PRINT):
        tp_frame = tp_frame[tp_frame != 0]
        fp_frame = fp_frame[fp_frame != 0]
        npos_frame = npos_frame[npos_frame != 0]
        tp_idx = tp_frame.nonzero()
        fp_idx = fp_frame.nonzero()
        npos_idx = npos_frame.nonzero()
        print('tp')
        print(tp_frame)
        print(tp_idx)
        print('fp')
        print(fp_frame)
        print(fp_idx)
        print('npos')
        print(npos_frame)
        print(npos_idx)
    map = mrec = mprec = 0
    prec = 0
    rec  = 0
    fp_sum = np.cumsum(fp, axis=0)
    tp_sum = np.cumsum(tp, axis=0)
    #fn     = 1-fp
    #fn_sum = np.cumsum(fn, axis=0)
    npos_sum = np.sum(npos, axis=0)
    #Override to avoid NaN
    if(npos_sum == 0):
        npos_sum = np.sum([1])
    #print('Difficulty Level: {:d}, fp sum: {:f}, tp sum: {:f} npos: {:d}'.format(i, fp_sum[i], tp_sum[i], npos[i]))
    #recall
    #Per frame per class AP
    rec = tp_sum[:] / npos_sum.astype(float)
    prec = tp_sum[:] / np.maximum(tp_sum[:] + fp_sum[:], np.finfo(np.float64).eps)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth precision
    mprec = np.average(prec)
    mrec = np.average(rec)
    rec, prec = zip(*sorted(zip(rec, prec)))
    map = waymo_ap(rec, prec)
    return mrec, mprec, map

def eval_load_recs(frameset, frame_path, labels_filename, db, mode, classname):
    class_recs = []
    filename = os.path.join(db._devkit_path, mode, 'labels',labels_filename)
    with open(filename,'r') as labels_file:
        data = labels_file.read()
        #print(data)
        labels = json.loads(data)
        for i, filename in enumerate(frameset):
            frame_idx = re.sub('[^0-9]','',filename)
            #Load annotations for frame, cut any elements not in classname
            tmp_rec = eval_load_rec(labels,frame_path,frame_idx,filename,db,mode)
            if(tmp_rec is None):
                tmp_rec = {}
                print('skipping frame {}, it does not exist in the ROIDB'.format(filename))
                tmp_rec['ignore_frame'] = True
            elif(tmp_rec['boxes'].size == 0):
                print('skipping frame {}, as it has no GT boxes'.format(filename))
                tmp_rec['ignore_frame'] = True
            else:
                tmp_rec['ignore_frame'] = False
                if(len(tmp_rec['gt_classes']) > 0):
                    gt_class_idx = np.where(tmp_rec['gt_classes'] == db._class_to_ind[classname])
                else:
                    gt_class_idx = []
                tmp_rec['gt_classes'] = tmp_rec['gt_classes'][gt_class_idx]
                tmp_rec['boxes'] = tmp_rec['boxes'][gt_class_idx]
                tmp_rec['gt_overlaps'] = tmp_rec['gt_overlaps'][gt_class_idx]
                tmp_rec['det'] = tmp_rec['det'][gt_class_idx]
                tmp_rec['ignore'] = tmp_rec['ignore'][gt_class_idx]
                tmp_rec['scene_idx'] = tmp_rec['scene_idx']
                tmp_rec['scene_desc'] = tmp_rec['scene_desc']
                #tmp_rec['difficulty'] = tmp_rec['difficulty'][gt_class_idx]
            tmp_rec['filename'] = filename
            tmp_rec['frame_idx']   = int(int(frame_idx)/cfg.MAX_IMG_PER_SCENE)
            tmp_rec['idx'] = frame_idx
            #List of all frames with GT boxes for a specific class
            class_recs.append(tmp_rec)
            #Only print every hundredth annotation?
            if i % 10 == 0:
                #print(recs[idx_name])
                print('Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(frameset)))
    return class_recs


def eval_load_rec(labels,frame_path,frame_idx,frame_file,db,mode='test'):
    tmp_rec = None
    for label in labels:
        #print(img_labels['assoc_frame'])
        if(label['assoc_frame'] == frame_idx):
            frame = os.path.join(frame_path,frame_file)
            #TODO: filter if not in preferred scene
            #TODO: here is where I get my ROI for the frame. Can use this as well as the variance to get average entropy
            #Do I really want multiple paths in which ROI's can be loaded? I should really fetch this from the existing ROIDB
            tmp_rec = db._load_waymo_annotation(frame,label,remove_without_gt=False,tod_filter_list=cfg.TEST.TOD_FILTER_LIST)
            break
    return tmp_rec


def eval_iou(bbgt,bbdet,eval_type):
    if(eval_type == '2d' or eval_type == 'bev_aa'):
        overlaps = eval_2d_iou(bbgt,bbdet)
    elif(eval_type == 'bev'):
        overlaps = eval_bev_iou(bbgt,bbdet)
    elif(eval_type == '3d'):
        overlaps = eval_3d_iou(bbgt,bbdet)
    else:
        overlaps = None
    return overlaps

"""
function: eval_2d_iou
Inputs:
bbgt: (N,4) ground truth boxes
bbdet: (4,) detection box
output:
overlaps: (N) total overlap for each bbgt with the bbdet
"""
def eval_2d_iou(bbgt,bbdet):
    # compute overlaps
    # intersection
    #bbgt = [xmin,ymin,xmax,ymax]
    ixmin = np.maximum(bbgt[:, 0], bbdet[0])
    iymin = np.maximum(bbgt[:, 1], bbdet[1])
    ixmax = np.minimum(bbgt[:, 2], bbdet[2])
    iymax = np.minimum(bbgt[:, 3], bbdet[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    #This is the intersection of both BB's
    inters = iw * ih

    # union
    uni = ((bbdet[2] - bbdet[0] + 1.) * (bbdet[3] - bbdet[1] + 1.) +
            (bbgt[:, 2] - bbgt[:, 0] + 1.) *
            (bbgt[:, 3] - bbgt[:, 1] + 1.) - inters)
    #IoU - intersection over union
    overlaps = inters / uni
    return overlaps


"""
function: eval_bev_iou
Inputs:
bbgt: (N,7) ground truth boxes
bbdet: (7,) detection box
output:
overlaps: (N) total overlap for each bbgt with the bbdet
"""
def eval_bev_iou(bbgt, bbdet):
    det_4pt = bbox_utils.bbox_3d_to_bev_4pt(bbdet[np.newaxis,:])[0]
    gts_4pt  = bbox_utils.bbox_3d_to_bev_4pt(bbgt)
    overlaps = np.zeros((bbgt.shape[0]))
    for i, gt_4pt in enumerate(gts_4pt):
        gt_poly = bbox_to_polygon_2d(gt_4pt)
        det_poly = bbox_to_polygon_2d(det_4pt)
        inter = gt_poly.intersection(det_poly).area
        union = gt_poly.union(det_poly).area
        iou_2d = inter/union
        overlaps[i] = iou_2d
    return overlaps

"""
function: eval_bev_iou
Inputs:
bbgt: (N,7) ground truth boxes
bbdet: (7,) detection box
output:
overlaps: (N) total overlap for each bbgt with the bbdet
"""
def eval_3d_iou(bbgt, bbdet):
    overlaps = np.zeros((bbgt.shape[0]))
    det_4pt = bbox_utils.bbox_3d_to_bev_4pt(bbdet[np.newaxis,:])[0]
    gts_4pt  = bbox_utils.bbox_3d_to_bev_4pt(bbgt)
    det_z    = [bbdet[2]-bbdet[5]/2,bbdet[2]+bbdet[5]/2]
    gt_z     = [bbgt[:,2]-bbgt[:,5]/2,bbgt[:,2]+bbgt[:,5]/2]
    det_height = bbdet[5]
    for i, gt_4pt in enumerate(gts_4pt):
        gt_height = bbgt[i,5]
        inter_max = min(gt_z[1][i],det_z[1]) 
        inter_min = max(gt_z[0][i],det_z[0])
        inter_height = max(0.0,inter_max - inter_min)
        gt_poly = bbox_to_polygon_2d(gt_4pt)
        det_poly = bbox_to_polygon_2d(det_4pt)
        inter = gt_poly.intersection(det_poly).area
        union = gt_poly.union(det_poly).area
        inter_vol = inter*inter_height
        #Compute iou 3d by including heights, as height is axis aligned
        iou_3d = inter_vol/(gt_poly.area*gt_height + det_poly.area*det_height - inter_vol)
        overlaps[i] = iou_3d
    return overlaps

def bbox_to_polygon_2d(bbox):
    return Polygon([(bbox[0,0], bbox[0,1]), (bbox[1,0], bbox[1,1]), (bbox[2,0], bbox[2,1]), (bbox[3,0], bbox[3,1])])

