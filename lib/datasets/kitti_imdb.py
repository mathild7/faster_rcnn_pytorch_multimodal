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
from datasets.imdb import imdb
# import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
# import scipy.io as sio
import pickle
import subprocess
import uuid
from .kitti_eval import kitti_eval
from model.config import cfg


class kitti_imdb(imdb):
    def __init__(self, mode='test'):
        name = 'kitti'
        imdb.__init__(self, name)
        self._year = None
        self._devkit_path = self._get_default_path()
        self._data_path = self._devkit_path
        self._mode = mode
        self._image_ext = '.png'
        self._image_sub_dir = 'image_2'
        if 'train' in mode:
            self._mode_sub_folder = 'training'
        elif 'eval' in mode or 'draw' in mode:
            self._mode_sub_folder = 'evaluation'
        elif 'test' in mode:
            self._mode_sub_folder = 'testing'
        elif 'exp' in mode:
            self._mode_sub_folder = 'experiment'
        self._classes = (
            'dontcare',  # always index 0
            'Pedestrian',
            'Car',
            'Cyclist')

        self.config = {
            'cleanup': True,
            'matlab_eval': False,
            'rpn_file': None
        }
        self._class_to_ind = dict(
            list(zip(self.classes, list(range(self.num_classes)))))
        self._image_dir = os.path.join(self._data_path, self._mode_sub_folder, 'image_2')
        self._image_index = sorted([d.replace('.png', '') for d in os.listdir(self._image_dir) if d.endswith('.png')])  
        #Limiter
        #self._image_index = self._image_index[:100]
        # Default to roidb handler
        self._roidb_handler = self.gt_roidb
        self._comp_id = 'comp4'

        assert os.path.exists(self._devkit_path), \
            'Kitti dataset path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
    Return the absolute path to image i in the image sequence.
    """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
    Construct an image path from the image's "index" identifier.
    """
        image_path = os.path.join(self._data_path, self._mode_sub_folder, self._image_sub_dir,
                                  index + '.png')
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def _get_default_path(self):
        """
    Return the default path where PASCAL VOC is expected to be installed.
    """
        return os.path.join(cfg.DATA_DIR, 'Kitti')

    def gt_roidb(self):
        """
    Return the database of ground-truth regions of interest.

    This function loads/saves from/to a cache file to speed up future calls.
    """
        cache_file = os.path.join(self.cache_path, self.name + '_' + self._mode + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                try:
                    roidb = pickle.load(fid)
                except:
                    roidb = pickle.load(fid, encoding='bytes')
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [
            self._load_kitti_annotation(index) for index in self.image_index
        ]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def get_class(self,idx):
       return self._classes[idx]

    def rpn_roidb(self):
        if int(self._year) == 2007 or self._mode_sub_folder != 'testing':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print('loading {}'.format(filename))
        assert os.path.exists(filename), \
            'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = pickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    #Only care about foreground classes
    def _load_kitti_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, self._mode_sub_folder, 'label_2', index + '.txt')
        label_lines = open(filename, 'r').readlines()
        objects = []
        num_objs = len(label_lines)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        ix = 0
        populated_idx = []
        for line in label_lines:
            label_arr = line.split(' ')
            # Make pixel indexes 0-based
            x1 = float(label_arr[4])
            y1 = float(label_arr[5])
            x2 = float(label_arr[6])
            y2 = float(label_arr[7])
            if(label_arr[0].strip() not in self._classes):
                #print('replacing {:s} with dont care'.format(label_arr[0]))
                label_arr[0] = 'dontcare'
            if('dontcare' not in label_arr[0].lower().strip()):
                #print(label_arr)
                cls = self._class_to_ind[label_arr[0].strip()]
                boxes[ix, :] = [x1, y1, x2, y2]
                gt_classes[ix] = cls
                overlaps[ix, cls] = 1.0
                seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)
                ix = ix + 1

        overlaps = scipy.sparse.csr_matrix(overlaps)
        #assert(len(boxes) != 0, "Boxes is empty for label {:s}".format(index))
        return {
            'boxes': boxes[0:ix],
            'gt_classes': gt_classes[0:ix],
            'gt_overlaps': overlaps[0:ix],
            'flipped': False,
            'seg_areas': seg_areas[0:ix]
        }

    def _get_comp_id(self):
        comp_id = (self._comp_id)
        return comp_id

    def _get_kitti_results_file_template(self):
        # data/Kitti/results/<comp_id>_test_aeroplane.txt
        filename = self._get_comp_id(
        ) + '_det_' + self._mode_sub_folder + '_{:s}.txt'
        path = os.path.join(self._devkit_path, 'results', filename)
        return path

    def _write_kitti_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == 'dontcare' or cls == '__background__':
                continue
            print('Writing {} KITTI results file'.format(cls))
            filename = self._get_kitti_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                #f.write('test')
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    #print('index: ' + index)
                    #print(dets)
                    if dets == []:
                        continue
                    # expects 1-based indices
                    for k in range(dets.shape[0]):
                        f.write(
                            '{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(
                                index, dets[k, -1], dets[k, 0],
                                dets[k, 1], dets[k, 2],
                                dets[k, 3]))

    def _do_python_eval(self, output_dir='output'):
        #annopath is for labels and only labelled images
        annopath = os.path.join(self._devkit_path, self._mode_sub_folder, 'label_2')
        print(annopath)
        #Not needed anymore, self._image_index has all files
        #imagesetfile = os.path.join(self._devkit_path, self._mode_sub_folder + '.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        aps = np.zeros((len(self._classes)-1,3))
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        #Loop through all classes
        for i, cls in enumerate(self._classes):
            if cls == 'dontcare'  or cls == '__background__':
                continue
            if 'Car' in cls:
                ovt = 0.7
            else:
                ovt = 0.5
            #Kitti/results/comp_X_testing_class.txt
            detfile = self._get_kitti_results_file_template().format(cls)
            #Run kitti evaluation metrics on each image
            rec, prec, ap = kitti_eval(
                detfile,
                annopath,
                self._image_index,
                cls,
                cachedir,
                ovthresh=ovt)
            aps[i-1,:] = ap
            #Tell user of AP
            print(('AP for {} = E {:.4f} M {:.4f} H {:.4f}'.format(cls,ap[0],ap[1],ap[2])))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print(('Mean AP = E {:.4f} M {:.4f} H {:.4f}'.format(np.mean(aps[:,0]),np.mean(aps[:,1]),np.mean(aps[:,2]))))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print(('E {:.3f} M {:.3f} H {:.3f}'.format(ap[0],ap[1],ap[2])))
        print(('Mean AP = E {:.4f} M {:.4f} H {:.4f}'.format(np.mean(aps[:,0]),np.mean(aps[:,1]),np.mean(aps[:,2]))))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

    def _do_matlab_eval(self, output_dir='output'):
        print('-----------------------------------------------------')
        print('Computing results with the official MATLAB eval code.')
        print('-----------------------------------------------------')
        path = os.path.join(cfg.ROOT_DIR, 'lib', 'datasets',
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'kitti_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
            .format(self._devkit_path, self._get_comp_id(), self._mode_sub_folder, output_dir)
        print(('Running:\n{}'.format(cmd)))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir):
        print('writing results to file...')
        self._write_kitti_results_file(all_boxes)
        self._do_python_eval(output_dir)
        if self.config['matlab_eval']:
            self._do_matlab_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == 'dontcare'  or cls == '__background__':
                    continue
                filename = self._get_kitti_results_file_template().format(cls)
                os.remove(filename)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True


if __name__ == '__main__':
    # from datasets.pascal_voc import pascal_voc
    #d = pascal_voc('trainval', '2007')
    #res = d.roidb
    from IPython import embed

    embed()
