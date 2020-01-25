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
import pickle
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import sys
import operator
import json

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
               imdb,
               imageset,
               classname,
               cachedir,
               mode,
               ovthresh=0.5):
    #Min overlap is 0.7 for cars, 0.5 for ped/bike
    """rec, prec, ap = waymo_eval(detpath,
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
    cachefile = os.path.join(cachedir, '{}_{}_annots.pkl'.format(mode,classname))
    # read list of images
    imagenames = imageset
    #if not os.path.isfile(cachefile):
        # load annotations
    class_recs = []
    for i, img in enumerate(imageset):
        #Load annotations for image, cut any elements not in classname
        labels_filename = os.path.join(imdb._devkit_path, mode, 'labels/labels.json')
        with open(labels_filename,'r') as labels_file:
            data = labels_file.read()
            #print(data)
            labels = json.loads(data)
            for img_labels in labels:
                #print(img_labels['assoc_frame'])
                if(img_labels['assoc_frame'] == img.replace('.png','')):
                    img = os.path.join(mode,'images',img)
                    #TODO: filter if not in preferred scene
                    #TODO: here is where I get my ROI for the image. Can use this as well as the variance to get average entropy
                    #Do I really want multiple paths in which ROI's can be loaded? I should really fetch this from the existing ROIDB
                    tmp_rec = imdb._load_waymo_annotation(img,img_labels,remove_without_gt=False,tod_filter_list=cfg.TEST.TOD_FILTER_LIST)
                    break
        if(tmp_rec is None):
            tmp_rec = {}
            print('skipping image {}, it does not exist in the ROIDB'.format(img))
            tmp_rec['imgname'] = img
            tmp_rec['ignore_img'] = True
        elif(tmp_rec['boxes'].size == 0):
            print('skipping image {}, as it has no GT boxes'.format(img))
            tmp_rec['ignore_img'] = True
        else:
            tmp_rec['ignore_img'] = False
            if(len(tmp_rec['gt_classes']) > 0):
                gt_class_idx = np.where(tmp_rec['gt_classes'] == imdb._class_to_ind[classname])
            else:
                gt_class_idx = []
            tmp_rec['gt_classes'] = tmp_rec['gt_classes'][gt_class_idx]
            tmp_rec['boxes'] = tmp_rec['boxes'][gt_class_idx]
            tmp_rec['gt_overlaps'] = tmp_rec['gt_overlaps'][gt_class_idx]
            tmp_rec['det'] = tmp_rec['det'][gt_class_idx]
            tmp_rec['ignore'] = tmp_rec['ignore'][gt_class_idx]
            tmp_rec['scene_idx'] = tmp_rec['scene_idx']
            tmp_rec['scene_desc'] = tmp_rec['scene_desc']
        #List of all images with GT boxes for a specific class
        class_recs.append(tmp_rec)
        #Only print every hundredth annotation?
        if i % 10 == 0:
            #print(recs[idx_name])
            print('Reading annotation for {:d}/{:d}'.format(
                i + 1, len(imageset)))
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
    image_idx = [x[0] for x in splitlines] #TODO: I dont like how this is along many images
    image_tokens = [x[1] for x in splitlines]
    confidence = np.array([float(x[2]) for x in splitlines])
    #All detections for specific class
    BB = np.array([[float(z) for z in x[3:]] for x in splitlines])
    #TODO: Add variance read here
    BB_var = np.array([[float(z) for z in x[7:]] for x in splitlines])
    #Repeated for X detections along every image presented
    idx = len(image_idx)
    #3 types, easy medium hard
    tp = np.zeros((idx))
    fp = np.zeros((idx))
    fn = np.zeros((idx))
    npos = np.zeros((len(class_recs)))
    for i, rec in enumerate(class_recs):
        if(rec['ignore_img'] is False):
            for ignore_elem in rec['ignore']:
                if(not ignore_elem):
                    npos[i] += 1
    if BB.shape[0] > 0:
        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        #No need to sort BB if you just access via sorted_ind
        #BB = BB[sorted_ind, :]
        image_idx_sorted = [int(image_idx[x]) for x in sorted_ind]
        image_tokens_sorted = [image_tokens[x] for x in sorted_ind]
        #print(image_ids)

        avg_scene_var  = np.zeros((cfg.NUM_SCENES,cfg.MAX_IMG_PER_SCENE))
        img_det_cnt    = np.zeros((cfg.NUM_SCENES,cfg.MAX_IMG_PER_SCENE))
        scene_desc     = ["" for x in range(cfg.NUM_SCENES)]

        # go down dets and mark true positives and false positives
        #Zip together sorted_ind with image tokens sorted. 
        #sorted_ind -> Needed to know which detection we are selecting next
        #image_tokens_sorted -> Needed to know which set of GT's are for the same image as the det
        print('num dets {}'.format(len(sorted_ind)))
        for det_idx,token in zip(sorted_ind,image_tokens_sorted):
            #R is a subset of detections for a specific class
            #print('doing det for image {}'.format(image_idx[d]))
            #Need to find associated GT image ID alongside its detection id 'd'
            #Only one such image, why appending?
            R = None
            skip_iter = False
            for rec in class_recs:
                #TODO: Janky fix...
                if(rec['imgname'].replace('val/images/','') == token):
                    if(rec['ignore_img'] is False):
                        R = rec
                    else:
                        skip_iter = True
                    break
            if(skip_iter):
                continue
            #Deprecated
            #R = class_recs[image_ids[d]]
            bb = BB[det_idx, :].astype(float)
            #Variance extraction, collect on a per scene basis
            bb_var = BB_var[det_idx, :].astype(float)
            avg_scene_var[R['scene_idx']][R['img_idx']] += np.average(bb_var)
            #print('setting mask to true for: {}-{}-{}'.format(R['scene_idx'],R['img_idx'],det_idx))
            img_det_cnt[R['scene_idx']][R['img_idx']] += 1
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
                #print('performing dont care filtering')
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
                #if ovmax > ovthresh:
                #ignore if not contained within easy, medium, hard
                if not R['ignore'][jmax]:
                    if not R['hit'][jmax]:
                        tp[det_idx] += 1
                        R['hit'][jmax] = True
                    else:
                        #If it already exists, cant double classify on same spot.
                        fp[det_idx] += 1
            #If your IoU is less than required, its simply a false positive.
            elif(BBGT.size > 0 and ovmax_dc < ovthresh_dc):
                #elif(BBGT.size > 0)
                fp[det_idx] += 1
        else:
            print('waymo eval, no GT boxes detected ')

    for scene_idx, scene_var in enumerate(avg_scene_var):
        if(scene_desc[scene_idx] != ""):
            scene_det_cnt = img_det_cnt[scene_idx]
            img_mask = np.array(scene_det_cnt,dtype=bool)
            avg_var = np.sum(scene_var)/np.sum(scene_det_cnt)
            #avg_var = np.average(scene_var[img_mask[scene_idx]])
            print('Scene: {} \n    num images {} \n     Description {} \n     Average Variance {:.2f}\n    total dets {}'.format(scene_idx,np.sum(img_mask),scene_desc[scene_idx],avg_var,np.sum(scene_det_cnt)))
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
    #Per image per class AP
    rec = tp_sum[:] / npos_sum.astype(float)
    prec = tp_sum[:] / np.maximum(tp_sum[:] + fp_sum[:], np.finfo(np.float64).eps)
    #if(i == 2):
    #    print(tp_sum[-1])
    #    print(fp_sum[-1])
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth precision
    mprec = np.average(prec)
    mrec = np.average(rec)
    rec, prec = zip(*sorted(zip(rec, prec)))
    map = waymo_ap(rec, prec)
    return mrec, mprec, map
