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
from model.config import cfg, get_output_dir
import shutil


class db(object):
    """database base class"""

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

    #TODO: Modify for any modality
    def _get_widths(self):
        return [
            PIL.Image.open(self.path_at(i)).size[0]
            for i in range(self.num_images)
        ]

    def get_class(self,idx):
       return self._classes[idx]
       
    def path_at(self, i, mode='train'):
        """
    Return the absolute path to image i in the image sequence.
    """
        if(mode == 'train'):
            return self.path_from_index(mode, self._train_index[i])
        elif(mode == 'val'):
            return self.path_from_index(mode, self._val_index[i])
        elif(mode == 'test'):
            return self.path_from_index(mode, self._test_index[i])
        else:
            return None
            
    def index_at(self, i, mode='train'):
        if(mode == 'train'):
            return self._train_index[i]
        elif(mode == 'val'):
            return self._val_index[i]
        elif(mode == 'test'):
            return self._test_index[i]
        else:
            return None

    def find_gt_for_frame(self,filename,mode):
        if(mode == 'train'):
            roidb = self.roidb
        elif(mode == 'val'):
            roidb = self.val_roidb
        for roi in roidb:
            if(roi['filename'] == filename):
                return roi
        return None

    def delete_eval_draw_folder(self,im_folder,mode):
        datapath = os.path.join(get_output_dir(self),'{}_drawn'.format(mode))
        #datapath = os.path.join(cfg.DATA_DIR, self._name ,im_folder,'{}_drawn'.format(mode))
        if(os.path.isdir(datapath)):
            print('deleting files in dir {}'.format(datapath))
            shutil.rmtree(datapath)
        os.makedirs(datapath)


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
