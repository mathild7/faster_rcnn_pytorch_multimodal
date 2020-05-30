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
import pickle
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import sys
import operator

#Values    Name      Description
#----------------------------------------------------------------------------
#   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
#                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
#                     'Misc' or 'DontCare'
#   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
#                     truncated refers to the object leaving image boundaries
#   1    occluded     Integer (0,1,2,3) indicating occlusion state:
#                     0 = fully visible, 1 = partly occluded
#                     2 = largely occluded, 3 = unknown
#   1    alpha        Observation angle of object, ranging [-pi..pi]
#   4    bbox         2D bounding box of object in the image (0-based index):
#                     contains left, top, right, bottom pixel coordinates
#   3    dimensions   3D object dimensions: height, width, length (in meters)
#   3    location     3D object location x,y,z in camera coordinates (in meters)
#   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
#   1    score        Only for results: Float, indicating confidence in
#                     detection, needed for p/r curves, higher is better.


def parse_rec(filename):
    """ Parse the labels for the specific image """
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


def kitti_ap(rec, prec):
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

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def kitti_eval_old(detpath,
             annopath,
             imageset,
             classname,
             cachedir,
             image_dir='label_2',
             ovthresh=0.5):
    #Min overlap is 0.7 for cars, 0.5 for ped/bike
    """rec, prec, ap = kitti_eval(detpath,
                              annopath,
                              imagesetfile,
                              classname,
                              [ovthresh])

  Top level function that does the PASCAL VOC evaluation.

  detpath: Path to detections
      detpath.format(classname) should produce the detection results file.
  annopath: Path to annotations
      annopath.format(imagename) should be the xml annotations file.
  imagesetfile: Text file containing the list of images, one image per line.
  classname: Category name (duh)
  cachedir: Directory for caching the annotations
  [ovthresh]: Overlap threshold (default = 0.5)
  [use_07_metric]: Whether to use VOC07's 11 point AP computation
      (default False)
  """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file
    idx = 0
    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, '%s_annots.pkl' % image_dir)
    # read list of images
    imagenames = imageset
    idx_names = []
    for image in imageset:
        idx_names.append(image)
    if not os.path.isfile(cachefile):
        # load annotations
        recs = {}
        for i, idx_name in enumerate(idx_names):
            recs[idx_name] = parse_rec(annopath + '/' + idx_names[i] + '.txt')
            #Only print every hundredth annotation?
            if i % 10 == 0:
                #print(recs[idx_name])
                print('Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(idx_names)))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        print('loading cached annotations from {:s}'.format(cachefile))
        with open(cachefile, 'rb') as f:
            try:
                recs = pickle.load(f)
            except:
                recs = pickle.load(f, encoding='bytes')

    # extract gt objects for this class
    class_recs = {}
    for idx_name in idx_names:
        #print('index: {:s}'.format(idx_name))
        #print(recs[idx_name])
        R = [obj for obj in recs[idx_name] if obj['name'] == classname]
        R_dc = [obj for obj in recs[idx_name] if obj['name'] == 'DontCare']
        bbox = np.array([x['bbox'] for x in R])
        bbox_dc = np.array([x['bbox'] for x in R_dc])
        #if use_diff:
        #    difficult = np.array([False for x in R]).astype(np.bool)
        #else:
        occ   = np.array([x['occluded'] for x in R])
        trunc = np.array([x['truncated'] for x in R])
        det = [False] * len(R)
        ignore = [True] * len(R)
        diff_e = [False] * len(R)
        diff_m = [False] * len(R)
        diff_h = [False] * len(R)
        #npos = npos + sum(~difficult)
        class_recs[idx_name] = {
            'bbox': bbox,
            'bbox_dc': bbox_dc,
            'occlusion': occ,
            'truncated': trunc,
            'hit': det,
            #dont ignore any ROIs for now
            'easy': diff_e,
            'medium': diff_m,
            'hard': diff_h,
            'ignore': ignore
        }
    ovthresh_dc = 0.5
    # read dets
    detfile = detpath.format(classname)
    print('Opening det file: ' + detfile)
    #sys.exit('donezo')
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    #All detections for specific class
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    #Repeated for X detections along every image presented
    idx = len(image_ids)
    #3 types, easy medium hard
    tp = np.zeros((idx, 3))
    fp = np.zeros((idx, 3))
    npos = np.zeros((idx, 3))
    if BB.shape[0] > 0:
        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]
        #print(image_ids)

        # go down dets and mark true positives and false positives
        for d in range(idx):
            #R is a subset of detections for a specific class
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            cat = []
            #Multiple possible bounding boxes, perhaps for multi car detection
            BBGT = R['bbox'].astype(float)
            BBGT_dc = R['bbox_dc'].astype(float)
            jmax = 0
            #Difficulty category
            #HARD - occlusion level -> 3, max trunc -> 50% max BB-H -> 25px
            for i, BBGT_elem in enumerate(BBGT):
                BBGT_height = BBGT_elem[3] - BBGT_elem[1]
                if(R['ignore'][i] is True):
                    if(R['occlusion'][i].astype(int) <= 2 and R['truncated'][i].astype(float) <= 0.5 and (BBGT_height) >= 25):
                        R['hard'][i] = True
                        R['ignore'][i] = False
                        npos[d, 2] += 1
                    if(R['occlusion'][i].astype(int) <= 1 and R['truncated'][i].astype(float) <= 0.3 and (BBGT_height) >= 25):
                        R['medium'][i] = True
                        R['ignore'][i] = False
                        npos[d, 1] += 1
                    if(R['occlusion'][i].astype(int) <= 0 and R['truncated'][i].astype(float) <= 0.15 and (BBGT_height) >= 40):
                        R['easy'][i] = True
                        R['ignore'][i] = False
                        npos[d, 0] += 1
            ovmax_dc = 0
            if BBGT_dc.size > 0:
                ixmin_dc = np.maximum(BBGT_dc[:, 0], bb[0])
                iymin_dc = np.maximum(BBGT_dc[:, 1], bb[1])
                ixmax_dc = np.minimum(BBGT_dc[:, 2], bb[2])
                iymax_dc = np.minimum(BBGT_dc[:, 3], bb[3])
                iw_dc = np.maximum(ixmax_dc - ixmin_dc + 1., 0.)
                ih_dc = np.maximum(iymax_dc - iymin_dc + 1., 0.)
                #This is the intersection of both BB's
                inters_dc = iw_dc * ih_dc

                # union
                uni_dc = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                       (BBGT_dc[:, 2] - BBGT_dc[:, 0] + 1.) *
                       (BBGT_dc[:, 3] - BBGT_dc[:, 1] + 1.) - inters_dc)
                #IoU - intersection over union
                overlaps_dc = inters_dc / uni_dc
                ovmax_dc = np.max(overlaps_dc)
            if BBGT.size > 0:
                # compute overlaps
                # intersection
                #BBGT = [xmin,ymin,xmax,ymax]
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                #This is the intersection of both BB's
                inters = iw * ih

                # union
                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                       (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                       (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)
                #IoU - intersection over union
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                #Index of max overlap between a BBGT and BB
                jmax = np.argmax(overlaps)
                ovmax_bbgt_height = BBGT[jmax, 3] - BBGT[jmax, 1]
            # Minimum IoU Threshold for a true positive
            if ovmax > ovthresh and ovmax_dc < ovthresh_dc:
                #ignore if not contained within easy, medium, hard
                if not R['ignore'][jmax]:
                    if not R['hit'][jmax]:
                        if(R['easy'][jmax] is True):
                            tp[d][0] += 1
                        if(R['medium'][jmax] is True):
                            tp[d][1] += 1
                        if(R['hard'][jmax] is True):
                            tp[d][2] += 1
                        R['hit'][jmax] = True
                    else:
                        #If it already exists, cant double classify on same spot.
                        if(R['easy'][jmax] and bb[3] - bb[1] >= 25):
                            fp[d][0] += 1
                        if(R['medium'][jmax] and bb[3] - bb[1] >= 25):
                            fp[d][1] += 1
                        if(R['hard'][jmax] and bb[3] - bb[1] >= 25):
                            fp[d][2] += 1
            #If your IoU is less than required, its simply a false positive.
            elif(BBGT.size > 0 and ovmax_dc < ovthresh_dc):
                if(R['easy'][jmax] is True and bb[3] - bb[1] >= 25):
                    fp[d][0] += 1
                if(R['medium'][jmax] is True and bb[3] - bb[1] >= 25):
                    fp[d][1] += 1
                if(R['hard'][jmax] is True and bb[3] - bb[1] >= 25):
                    fp[d][2] += 1

    map = mrec = mprec = np.zeros(3)
    prec = []
    rec  = []
    fp_sum = np.cumsum(fp, axis=0)
    tp_sum = np.cumsum(tp, axis=0)
    #fn     = 1-fp
    #fn_sum = np.cumsum(fn, axis=0)
    npos_sum = np.sum(npos, axis=0)
    for i in range(0,3):
        #print('Difficulty Level: {:d}, fp sum: {:f}, tp sum: {:f} npos: {:d}'.format(i, fp_sum[i], tp_sum[i], npos[i]))
        #recall
        #Per image per class AP
        rec = tp_sum[:, i] / npos_sum[i].astype(float)
        prec = tp_sum[:, i] / np.maximum(tp_sum[:, i] + fp_sum[:, i], np.finfo(np.float64).eps)
        #if(i == 2):
        #    print(tp_sum[-1])
        #    print(fp_sum[-1])
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth precision
        mprec[i] = np.average(prec)
        mrec[i] = np.average(rec)
        rec, prec = zip(*sorted(zip(rec, prec)))
        map[i] = kitti_ap(rec, prec)
    return mrec, mprec, map
