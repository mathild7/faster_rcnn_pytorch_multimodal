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

    def __init__(self, name, mode='train', classes=None):
        self._name = name
        self._num_classes = 0
        if not classes:
            self._classes = []
        else:
            self._classes = classes
        self._obj_proposer = 'gt'
        self._roidb = None
        self._val_roidb = None
        # Use this dict for storing dataset specific config options
        self.config = {}
        self._train_index = []
        self._val_index = []
        self._test_index = []
        self._mode = mode
        print('db mode: {}'.format(mode))
        self._devkit_path = self._get_default_path()

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

    def _get_results_file_template(self, mode,class_name,output_dir):
        # data/waymo/results/<comp_id>_test_aeroplane.txt
        filename = 'det_' + mode + '_{:s}.txt'.format(class_name)
        result_dir = os.path.join(output_dir, 'results')
        if(not os.path.isdir(result_dir)):
            os.mkdir(result_dir)
        path = os.path.join(result_dir, filename)
        return path

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

    def mode_to_sub_folder(self,mode):
        if(mode == 'train'):
            return self._train_sub_folder
        elif(mode == 'val'):
            return self._val_sub_folder
        elif(mode == 'test'):
            return self._test_sub_folder
        else:
            return None

    def _get_index_for_mode(self, mode):
        if(mode == 'train'):
            return self._train_index
        elif(mode == 'val'):
            return self._val_index
        elif(mode == 'test'):
            return self._test_index
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
        datapath = os.path.join(get_output_dir(self,mode),'{}_drawn'.format(im_folder))
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

    def _find_draw_folder(self, mode, draw_folder):
        if(draw_folder is None):
            draw_folder = mode
        out_dir = os.path.join(get_output_dir(self,mode=mode),'{}_drawn'.format(draw_folder))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        return out_dir

    def _normalize_uncertainties(self,dets,uncertainties):
        normalized_uncertainties = {}
        for key,uc in uncertainties.items():
            if('bbox' in key):
                #stds = uc.data.new(cfg.TRAIN.IMAGE.BBOX_NORMALIZE_STDS).unsqueeze(0).expand_as(uc)
                #means = uc.data.new(cfg.TRAIN.IMAGE.BBOX_NORMALIZE_MEANS).unsqueeze(0).expand_as(uc)
                #uc = uc.mul(stds).add(means)
                #bbox_width  = dets[:,2] - dets[:,0]
                #bbox_height = dets[:,3] - dets[:,1]
                #bbox_size = np.sqrt(bbox_width*bbox_height)
                #uc[:,0] = uc[:,0]/bbox_size
                #uc[:,2] = uc[:,2]/bbox_size
                #uc[:,1] = uc[:,1]/bbox_size
                #uc[:,3] = uc[:,3]/bbox_size
                normalized_uncertainties[key] = np.mean(uc,axis=1)
            elif('mutual_info' in key):
                normalized_uncertainties[key] = uc.squeeze(1)  #*10*(-np.log(dets[:,4]))
            else:
                normalized_uncertainties[key] = uc.squeeze(1)
        return normalized_uncertainties

    def _sort_dets_by_uncertainty(self,dets,uncertainties,descending=False):
        if(cfg.UC.EN_BBOX_ALEATORIC and self._uncertainty_sort_type == 'a_bbox_var'):
            sortable = uncertainties['a_bbox_var']
        elif(cfg.UC.EN_BBOX_EPISTEMIC and self._uncertainty_sort_type == 'e_bbox_var'):
            sortable = uncertainties['e_bbox_var']
        elif(cfg.UC.EN_CLS_ALEATORIC and self._uncertainty_sort_type == 'a_entropy'):
            sortable = uncertainties['a_entropy']
        elif(cfg.UC.EN_CLS_ALEATORIC and self._uncertainty_sort_type == 'a_mutual_info'):
            sortable = uncertainties['a_mutual_info']
        elif(cfg.UC.EN_CLS_ALEATORIC and self._uncertainty_sort_type == 'a_cls_var'):
            sortable = uncertainties['a_cls_var']
        elif(cfg.UC.EN_CLS_EPISTEMIC and self._uncertainty_sort_type == 'e_mutual_info'):
            sortable = uncertainties['e_mutual_info']
        elif(cfg.UC.EN_CLS_EPISTEMIC and self._uncertainty_sort_type == 'e_entropy'):
            sortable = uncertainties['e_entropy']
        else:
            sortable = np.arange(0,dets.shape[0])
        if(descending is True):
            return np.argsort(-sortable)
        else:
            return np.argsort(sortable)

    def _write_image_results_file(self, all_boxes, output_dir, mode):
        img_idx = self._get_index_for_mode(mode)
        for cls_ind, cls in enumerate(self.classes):
            if cls == 'dontcare' or cls == '__background__':
                continue
            print('Writing {} {} results file'.format(self.name, cls))
            filename = self._get_results_file_template(mode,cls,output_dir)
            with open(filename, 'wt') as f:
                #f.write('test')
                for im_ind, img in enumerate(img_idx):
                    dets = all_boxes[cls_ind][im_ind]
                    #TODO: Add this to dets file
                    #dets_bbox_var = dets[0:4]
                    #dets = dets[4:]
                    #print('index: ' + index)
                    #print(dets)
                    if dets.size == 0:
                        continue
                    # expects 1-based indices
                    #TODO: Add variance to output file
                    for k in range(dets.shape[0]):
                        f.write(
                            '{:d} {:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}'.format(
                                im_ind, img, dets[k, 4], 
                                dets[k, 0], dets[k, 1], 
                                dets[k, 2], dets[k, 3]))
                        #Write uncertainties
                        for l in range(5,dets.shape[1]):
                            f.write(' {:.2f}'.format(dets[k,l]))
                        f.write('\n')

    def _write_lidar_results_file(self, all_boxes, output_dir, mode):
        frame_list = self._get_index_for_mode(mode)
        for cls_ind, cls in enumerate(self.classes):
            if cls == 'dontcare' or cls == '__background__':
                continue
            print('Writing {} {} results file'.format(self.name, cls))
            filename = self._get_results_file_template(mode,cls,output_dir)
            with open(filename, 'wt') as f:
                #f.write('test')
                for ind, frame in enumerate(frame_list):
                    dets = all_boxes[cls_ind][ind]
                    #TODO: Add this to dets file
                    #dets_bbox_var = dets[0:4]
                    #dets = dets[4:]
                    #print('index: ' + index)
                    #print(dets)
                    if dets.size == 0:
                        continue
                    # expects 1-based indices
                    #TODO: Add variance to output file
                    for k in range(dets.shape[0]):
                        f.write(
                            '{:d} {:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.2f} {:.2f} {:.2f} {:.3f}'.format(
                                ind, frame, dets[k, 7], 
                                dets[k, 0], dets[k, 1], 
                                dets[k, 2], dets[k, 3],
                                dets[k, 4], dets[k, 5], dets[k, 6]))
                        #Write uncertainties
                        if(dets.shape[1] > cfg.LIDAR.NUM_BBOX_ELEM+1):
                            for l in range(8,dets.shape[1]):
                                f.write(' {:.3f}'.format(dets[k,l]))
                        f.write('\n')
    #LIDAR specific functions
    def _transform_to_pixel_coords(self,coords,inv_x=False,inv_y=False):
        y = (coords[1]-cfg.LIDAR.Y_RANGE[0])*self._draw_height/(cfg.LIDAR.Y_RANGE[1] - cfg.LIDAR.Y_RANGE[0])
        x = (coords[0]-cfg.LIDAR.X_RANGE[0])*self._draw_width/(cfg.LIDAR.X_RANGE[1] - cfg.LIDAR.X_RANGE[0])
        if(inv_x):
            x = self._draw_width - x
        if(inv_y):
            y = self._draw_height - y
        return (int(x), int(y))

    def draw_bev(self,bev_img,draw):
        coord = []
        color = []
        for i,point in enumerate(bev_img):
            z_max = cfg.LIDAR.Z_RANGE[1]
            z_min = cfg.LIDAR.Z_RANGE[0]
            #Point is contained within slice
            #TODO: Ensure <= for all if's, or else elements right on the divide will be ignored
            if(point[2] >= z_min and point[2] < z_max):
                coords = self._transform_to_pixel_coords(point,inv_x=False,inv_y=False)
                c = int((point[2]-z_min)*255/(z_max - z_min))
                draw.point(coords, fill=(int(c),0,0))
        return draw

    def draw_bev_slice(self,bev_slice,bev_idx,draw):
        coord = []
        color = []
        for i,point in enumerate(bev_slice):
            z_max, z_min = self._slice_height(bev_idx)
            #Point is contained within slice
            #TODO: Ensure <= for all if's, or else elements right on the divide will be ignored
            if(point[2] < z_max or point[2] >= z_min):
                coords = self._transform_to_pixel_coords(point)
                c = int((point[2]-z_min)*255/(z_max - z_min))
                draw.point(coords, fill=int(c))
        return draw

    #DEPRECATED
    #def _sample_bboxes(self,softmax,entropy,bbox,bbox_var):
    #    sampled_det = np.zeros((5,cfg.UC.NUM_BBOX_SAMPLE))
    #    det_width = max(int((entropy)*10),-1)+2
    #    bbox_samples = np.random.normal(bbox,np.sqrt(bbox_var),size=(cfg.UC.NUM_BBOX_SAMPLE,7))
    #    sampled_det[0:4][:] = np.swapaxes(bbox_samples,1,0)
    #    sampled_det[4][:] = np.repeat(softmax,cfg.UC.NUM_BBOX_SAMPLE)
    #    return sampled_det
    
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
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True
