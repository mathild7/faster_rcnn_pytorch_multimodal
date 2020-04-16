# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import PIL
from utils.bbox import bbox_overlaps
import numpy as np
import scipy.sparse
from torchvision.ops import nms
import torch
from model.config import cfg


class imdb(object):
    """Image database."""

    def __init__(self, name, classes=None):
        self._name = name
        self._num_classes = 0
        if not classes:
            self._classes = []
        else:
            self._classes = classes
        self._image_index = []
        self._obj_proposer = 'gt'
        self._roidb = None
        self._val_roidb = None
        # Use this dict for storing dataset specific config options
        self.config = {}

    @property
    def name(self):
        return self._name

    @property
    def num_classes(self):
        return len(self._classes)

    @property
    def classes(self):
        return self._classes

    @property
    def image_index(self):
        return self._image_index

    @property
    def roidb(self):
        # A roidb is a list of dictionaries, each with the following keys:
        #   boxes
        #   gt_overlaps
        #   gt_classes
        #   flipped
        if self._roidb is not None:
            return self._roidb
        self._roidb = self.gt_roidb('train')
        return self._roidb

    @property
    def val_roidb(self):
        # A roidb is a list of dictionaries, each with the following keys:
        #   boxes
        #   gt_overlaps
        #   gt_classes
        #   flipped
        if self._val_roidb is not None:
            return self._val_roidb
        self._val_roidb = self.gt_roidb('val')
        return self._val_roidb

    @property
    def cache_path(self):
        cache_path = osp.abspath(osp.join(cfg.ROOT_DIR, 'cache'))
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path

    @property
    def num_images(self):
        return len(self.image_index)

    def default_roidb(self):
        raise NotImplementedError

    def evaluate_detections(self, all_boxes, output_dir=None):
        """
    all_boxes is a list of length number-of-classes.
    Each list element is a list of length number-of-images.
    Each of those list elements is either an empty list []
    or a numpy array of detection.

    all_boxes[class][image] = [] or np.array of shape #dets x 5
    """
        raise NotImplementedError

    def _get_widths(self):
        return [
            PIL.Image.open(self.path_at(i)).size[0]
            for i in range(self.num_images)
        ]


    def path_at(self, i, mode='train'):
        """
    Return the absolute path to image i in the image sequence.
    """
        if(mode == 'train'):
            return self.image_path_from_index(mode, self._train_index[i])
        elif(mode == 'val'):
            return self.image_path_from_index(mode, self._val_index[i])
        elif(mode == 'test'):
            return self.image_path_from_index(mode, self._test_index[i])
        else:
            return None
            
    def image_index_at(self, i, mode='train'):
        if(mode == 'train'):
            return self._train_index[i]
        elif(mode == 'val'):
            return self._val_index[i]
        elif(mode == 'test'):
            return self._test_index[i]
        else:
            return None

    def append_flipped_images(self, mode):
        if(mode == 'train'):
            num_images = len(self._roidb)
        elif(mode == 'val'):
            num_images = len(self._val_roidb)
        for i in range(num_images):
            boxes = self.roidb[i]['boxes'].copy()
            boxes_dc = self.roidb[i]['boxes_dc'].copy()
            img_index = self.roidb[i]['imgname']
            filepath  = self.roidb[i]['filename']
            ignore    = self.roidb[i]['ignore'].copy()
            cat = self.roidb[i]['cat'].copy()
            oldx1 = boxes[:, 0].copy()
            oldx2 = boxes[:, 2].copy()
            boxes[:, 0] = self._imwidth - oldx2 - 1
            boxes[:, 2] = self._imwidth - oldx1 - 1
            oldx1_dc = boxes_dc[:, 0].copy()
            oldx2_dc = boxes_dc[:, 2].copy()
            boxes_dc[:, 0] = self._imwidth - oldx2_dc - 1
            boxes_dc[:, 2] = self._imwidth - oldx1_dc - 1
            assert (boxes[:, 2] >= boxes[:, 0]).all()
            assert (boxes_dc[:, 2] >= boxes_dc[:, 0]).all()
            entry = {
                'imgname': img_index,
                'filename': filepath,
                'cat': cat,
                'ignore': ignore,
                'boxes': boxes,
                'boxes_dc' : boxes_dc,
                'gt_overlaps': self.roidb[i]['gt_overlaps'],
                'gt_classes': self.roidb[i]['gt_classes'],
                'flipped': True
            }
            #Calls self.gt_roidb through a handler.
            if(mode == 'train'):
                self._roidb.append(entry)
            elif(mode == 'val'):
                self._val_roidb.append(entry)

    def evaluate_recall(self,
                        candidate_boxes=None,
                        thresholds=None,
                        area='all',
                        limit=None):
        """Evaluate detection proposal recall metrics.

    Returns:
        results: dictionary of results with keys
            'ar': average recall
            'recalls': vector recalls at each IoU overlap threshold
            'thresholds': vector of IoU overlap thresholds
            'gt_overlaps': vector of all ground-truth overlaps
    """
        # Record max overlap value for each gt box
        # Return vector of overlap values
        areas = {
            'all': 0,
            'small': 1,
            'medium': 2,
            'large': 3,
            '96-128': 4,
            '128-256': 5,
            '256-512': 6,
            '512-inf': 7
        }
        area_ranges = [
            [0**2, 1e5**2],  # all
            [0**2, 32**2],  # small
            [32**2, 96**2],  # medium
            [96**2, 1e5**2],  # large
            [96**2, 128**2],  # 96-128
            [128**2, 256**2],  # 128-256
            [256**2, 512**2],  # 256-512
            [512**2, 1e5**2],  # 512-inf
        ]
        assert area in areas, 'unknown area range: {}'.format(area)
        area_range = area_ranges[areas[area]]
        gt_overlaps = np.zeros(0)
        num_pos = 0
        for i in range(self.num_images):
            # Checking for max_overlaps == 1 avoids including crowd annotations
            # (...pretty hacking :/)
            max_gt_overlaps = self.roidb[i]['gt_overlaps'].toarray().max(
                axis=1)
            gt_inds = np.where((self.roidb[i]['gt_classes'] > 0) &
                               (max_gt_overlaps == 1))[0]
            gt_boxes = self.roidb[i]['boxes'][gt_inds, :]
            gt_areas = self.roidb[i]['seg_areas'][gt_inds]
            valid_gt_inds = np.where((gt_areas >= area_range[0]) &
                                     (gt_areas <= area_range[1]))[0]
            gt_boxes = gt_boxes[valid_gt_inds, :]
            num_pos += len(valid_gt_inds)

            if candidate_boxes is None:
                # If candidate_boxes is not supplied, the default is to use the
                # non-ground-truth boxes from this roidb
                non_gt_inds = np.where(self.roidb[i]['gt_classes'] == 0)[0]
                boxes = self.roidb[i]['boxes'][non_gt_inds, :]
            else:
                boxes = candidate_boxes[i]
            if boxes.shape[0] == 0:
                continue
            if limit is not None and boxes.shape[0] > limit:
                boxes = boxes[:limit, :]

            overlaps = bbox_overlaps(
                boxes.astype(np.float), gt_boxes.astype(np.float))

            _gt_overlaps = np.zeros((gt_boxes.shape[0]))
            for j in range(gt_boxes.shape[0]):
                # find which proposal box maximally covers each gt box
                argmax_overlaps = overlaps.argmax(axis=0)
                # and get the iou amount of coverage for each gt box
                max_overlaps = overlaps.max(axis=0)
                # find which gt box is 'best' covered (i.e. 'best' = most iou)
                gt_ind = max_overlaps.argmax()
                gt_ovr = max_overlaps.max()
                assert (gt_ovr >= 0)
                # find the proposal box that covers the best covered gt box
                box_ind = argmax_overlaps[gt_ind]
                # record the iou coverage of this gt box
                _gt_overlaps[j] = overlaps[box_ind, gt_ind]
                assert (_gt_overlaps[j] == gt_ovr)
                # mark the proposal box and the gt box as used
                overlaps[box_ind, :] = -1
                overlaps[:, gt_ind] = -1
            # append recorded iou coverage level
            gt_overlaps = np.hstack((gt_overlaps, _gt_overlaps))

        gt_overlaps = np.sort(gt_overlaps)
        if thresholds is None:
            step = 0.05
            thresholds = np.arange(0.5, 0.95 + 1e-5, step)
        recalls = np.zeros_like(thresholds)
        # compute recall for each iou threshold
        for i, t in enumerate(thresholds):
            recalls[i] = (gt_overlaps >= t).sum() / float(num_pos)
        # ar = 2 * np.trapz(recalls, thresholds)
        ar = recalls.mean()
        return {
            'ar': ar,
            'recalls': recalls,
            'thresholds': thresholds,
            'gt_overlaps': gt_overlaps
        }

    def create_roidb_from_box_list(self, box_list, gt_roidb):
        assert len(box_list) == self.num_images, \
          'Number of boxes must match number of ground-truth images'
        roidb = []
        for i in range(self.num_images):
            boxes = box_list[i]
            num_boxes = boxes.shape[0]
            overlaps = np.zeros((num_boxes, self.num_classes),
                                dtype=np.float32)

            if gt_roidb is not None and gt_roidb[i]['boxes'].size > 0:
                gt_boxes = gt_roidb[i]['boxes']
                gt_classes = gt_roidb[i]['gt_classes']
                gt_overlaps = bbox_overlaps(
                    boxes.astype(np.float), gt_boxes.astype(np.float))
                argmaxes = gt_overlaps.argmax(axis=1)
                maxes = gt_overlaps.max(axis=1)
                I = np.where(maxes > 0)[0]
                overlaps[I, gt_classes[argmaxes[I]]] = maxes[I]

            overlaps = scipy.sparse.csr_matrix(overlaps)
            roidb.append({
                'boxes':
                boxes,
                'gt_classes':
                np.zeros((num_boxes, ), dtype=np.int32),
                'gt_overlaps':
                overlaps,
                'flipped':
                False,
                'seg_areas':
                np.zeros((num_boxes, ), dtype=np.float32),
            })
        return roidb


    #TODO: Rework to accept all kinds of variance
    def nms_hstack_var(self,var_type,var,samples,inds,keep,c):
        if(var_type == 'cls'):
            cls_var = var[inds,c]
            cls_var = cls_var[keep,np.newaxis]
            return cls_var
        elif(var_type == 'bbox'):
            cls_bbox_var = var[inds, c * 4:(c + 1) * 4]
            cls_bbox_var = cls_bbox_var[keep, :]
            bbox_samples = samples[inds, c * 4:(c + 1) * 4]
            bbox_samples = bbox_samples[keep, :, :]
            return cls_bbox_var, bbox_samples
        else:
            return None


    #DEPRECATED
    #def nms_hstack(self,scores,mean_boxes,thresh,c):
    #    inds         = np.where(scores[:, c] > thresh)[0]
    #    #No detections over threshold
    #    if(inds.size == 0):
    #        print('no detections for image over threshold {}'.format(thresh))
    #        return np.empty(0),[],[]
    #    cls_scores   = scores[inds, c]
    #    cls_boxes    = mean_boxes[inds, c * 4:(c + 1) * 4]
    #    #[cls_var,cls_boxes,cls_scores]
    #    cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
    #        .astype(np.float32, copy=False)
    #    keep = nms(
    #        torch.from_numpy(cls_boxes.astype(np.float32)), torch.from_numpy(cls_scores),
    #        cfg.TEST.NMS).numpy() if cls_dets.size > 0 else []
    #    cls_dets = cls_dets[keep, :]
    #    #Only if this variable has been provided
    #    return cls_dets, inds, keep

    @staticmethod
    def merge_roidbs(a, b):
        assert len(a) == len(b)
        for i in range(len(a)):
            a[i]['boxes'] = np.vstack((a[i]['boxes'], b[i]['boxes']))
            a[i]['gt_classes'] = np.hstack((a[i]['gt_classes'],
                                            b[i]['gt_classes']))
            a[i]['gt_overlaps'] = scipy.sparse.vstack(
                [a[i]['gt_overlaps'], b[i]['gt_overlaps']])
            a[i]['seg_areas'] = np.hstack((a[i]['seg_areas'],
                                           b[i]['seg_areas']))
        return a

    def competition_mode(self, on):
        """Turn competition mode on or off."""
        pass
