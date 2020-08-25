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
from datasets.db import db
# import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import re
# import scipy.io as sio
import pickle
from PIL import Image, ImageDraw
import subprocess
import uuid
import traceback
from .cadc_eval import cadc_eval
from model.config import cfg, get_output_dir
import shutil
from random import SystemRandom
import csv

class cadc_imdb(db):
    def __init__(self, mode='test',limiter=0, shuffle_en=True):
        name = 'cadc'
        db.__init__(self, name, mode)
        self._devkit_path = self._get_default_path()
        self._data_path = self._devkit_path
        self._mode = mode
        self._uncertainty_sort_type = cfg.UC.SORT_TYPE
        self._frame_sub_dir = 'image_00'
        if(mode == 'test'):
            self._tod_filter_list = cfg.TEST.CADC_FILTER_LIST
        else:
            self._tod_filter_list = cfg.TRAIN.CADC_FILTER_LIST
        scene_desc_filename = os.path.join(self._data_path, 'cadc_scene_description.csv')
        self._load_scene_meta(scene_desc_filename)
        self._train_dir = os.path.join(self._data_path, 'train', self._frame_sub_dir)
        self._val_dir   = os.path.join(self._data_path, 'val', self._frame_sub_dir)
        #self._test_dir   = os.path.join(self._data_path, 'testing', self._frame_sub_dir)
        crop_top        = 150
        crop_bottom     = 250
        self._imwidth = 1280
        self._imheight = 1024 - crop_top - crop_bottom
        self._imtype = 'png'
        self._filetype = 'png'
        self.type = 'image'
        self._mode = mode
        #Backwards compatibility
        #self._train_sub_folder = 'training'
        #self._val_sub_folder = 'evaluation'
        #self._test_sub_folder = 'testing'
        self._classes = (
            'dontcare',  # always index 0
            #'Pedestrian',
            #'Cyclist',
            'Car')

        self.config = {
            'cleanup': True,
            'matlab_eval': False,
            'rpn_file': None
        }
        self._class_to_ind = dict(
            list(zip(self.classes, list(range(self.num_classes)))))
        self._train_index = sorted([d for d in os.listdir(self._train_dir) if d.endswith('.png')])
        self._val_index   = sorted([d for d in os.listdir(self._val_dir) if d.endswith('.png')])
        #self._test_index  = sorted([d for d in os.listdir(self._test_dir) if d.endswith('.png')])
        #Limiter
        if(limiter != 0):
            if(limiter < len(self._val_index)):
                self._val_index   = self._val_index[:limiter]
            if(limiter < len(self._train_index)):
                self._train_index = self._train_index[:limiter]
            #if(limiter < len(self._test_index)):
            #    self._test_index = self._test_index[:limiter]
        rand = SystemRandom()
        if(shuffle_en):
            print('shuffling image indices')
            rand.shuffle(self._val_index)
            rand.shuffle(self._train_index)
            #rand.shuffle(self._test_index)
        assert os.path.exists(self._devkit_path), \
            'cadc dataset path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)

    def path_from_index(self, mode, index):
        """
    Construct an image path from the image's "index" identifier.
    """
        image_path = os.path.join(self._devkit_path, mode, self._frame_sub_dir, index)
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def _get_default_path(self):
        """
    Return the default path where PASCAL VOC is expected to be installed.
    """
        return os.path.join(cfg.DATA_DIR, 'cadc')

    def gt_roidb(self, mode):
        """
    Return the database of ground-truth regions of interest.

    This function loads/saves from/to a cache file to speed up future calls.
    """
        #for line in traceback.format_stack():
        #    print(line.strip())
        cache_file = os.path.join(self._get_cache_dir(), self._name + '_' + mode + '_image_gt_roidb.pkl')
        image_index = self._get_index_for_mode(mode)
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                try:
                    roidb = pickle.load(fid)
                except:
                    roidb = pickle.load(fid, encoding='bytes')
            print('{} gt roidb loaded from {}'.format(self._name, cache_file))
            return roidb

        gt_roidb = []
        for img in image_index:
            idx = img.replace('.png','')
            roi = self._load_cadc_annotation(idx, mode)
            if(roi is not None):
                gt_roidb.append(roi)
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    #Only care about foreground classes
    def _load_cadc_annotation(self, index, mode='train', remove_without_gt=True):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        #print('loading cadc anno')
        filename = os.path.join(self._data_path, mode, 'annotation_00', index + '.txt')
        img_filename = os.path.join(self._data_path, mode, self._frame_sub_dir, index + '.png')
        label_lines = open(filename, 'r').readlines()
        num_objs = len(label_lines)
        boxes      = np.zeros((num_objs, cfg.IMAGE.NUM_BBOX_ELEM), dtype=np.uint16)
        boxes_dc   = np.zeros((num_objs, cfg.IMAGE.NUM_BBOX_ELEM), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        ignore     = np.zeros((num_objs), dtype=np.bool)
        gt_diff    = np.zeros((num_objs), dtype=np.int8)
        gt_trunc   = np.zeros((num_objs), dtype=np.float32)
        gt_occ     = np.zeros((num_objs), dtype=np.int16)
        gt_ids     = np.zeros((num_objs), dtype=np.int16)
        gt_pts     = np.zeros((num_objs), dtype=np.int16)
        gt_alpha   = np.zeros((num_objs), dtype=np.float32)
        gt_dist    = np.zeros((num_objs), dtype=np.float32)
        cat = []
        overlaps   = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas  = np.zeros((num_objs), dtype=np.float32)
        # Load object bounding boxes into a data frame.
        ix = 0
        ix_dc = 0
        populated_idx = []
        for line in label_lines:
            label_arr = line.split(' ')
            # Make pixel indexes 0-based
            x1 = float(label_arr[4])
            y1 = float(label_arr[5])
            x2 = float(label_arr[6])
            y2 = float(label_arr[7])
            BBGT_height = y2 - y1
            trunc = float(label_arr[1])
            occ   = int(label_arr[2])
            alpha = float(label_arr[3])
            BBGT_height = y2 - y1
            l_x = float(label_arr[8])
            w_y = float(label_arr[9])
            h_z = float(label_arr[10])
            x_c = float(label_arr[11])
            y_c = float(label_arr[12])
            z_c = float(label_arr[13])
            heading = float(label_arr[14].replace('\n',''))
            pts = int(label_arr[15])
            drive = re.sub('[^0-9]','', label_arr[16])
            scene = label_arr[17]
            scene_idx = int(drive)*100 + int(scene)
            scene_desc = self._get_scene_desc(scene_idx)
            if(scene_desc not in self._tod_filter_list):
                continue
            diff = 0
            if(BBGT_height < 10):
                label_arr[0] = 'dontcare'
            if(x_c > 60):
                continue
            if(pts < 10):
                continue
            #if(occ <= 0 and trunc <= 0.15 and (BBGT_height) >= 40):
            #    diff = 0
            #elif(occ <= 1 and trunc <= 0.3 and (BBGT_height) >= 25):
            #    diff = 1
            #elif(occ <= 2 and trunc <= 0.5 and (BBGT_height) >= 25):
            #    diff = 2
            if(diff == -1):
                label_arr[0] = 'dontcare'
            if(label_arr[0].strip() not in self._classes):
                #print('replacing {:s} with dont care'.format(label_arr[0]))
                label_arr[0] = 'dontcare'
            if('dontcare' not in label_arr[0].lower().strip()):
                #print(label_arr)
                cls_type = self._class_to_ind[label_arr[0].strip()]
                cat.append(label_arr[0].strip())
                boxes[ix, :] = [x1, y1, x2, y2]
                gt_classes[ix] = cls_type
                gt_trunc[ix] = trunc
                gt_occ[ix]   = occ
                gt_alpha[ix] = alpha
                gt_pts[ix] = pts
                gt_dist[ix] = x_c
                gt_ids[ix]   = int(index) + ix
                gt_diff[ix]  = diff
                #overlaps is (NxM) where N = number of GT entires and M = number of classes
                overlaps[ix, cls_type] = 1.0
                seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)
                ix = ix + 1
            if('dontcare' in label_arr[0].lower().strip()):
                #print(line)
                boxes_dc[ix_dc, :] = [x1, y1, x2, y2]
                ix_dc = ix_dc + 1
                

        overlaps = scipy.sparse.csr_matrix(overlaps)
        #assert(len(boxes) != 0, "Boxes is empty for label {:s}".format(index))
        if(ix == 0 and remove_without_gt is True):
            print('removing img {} with no GT boxes specified'.format(index))
            return None
        return {
            'idx': index,
            'scene_idx': scene_idx,
            'filename': img_filename,
            'det': ignore[0:ix].copy(),
            'ignore':ignore[0:ix],
            'hit': ignore[0:ix].copy(),
            'trunc': gt_trunc[0:ix],
            'occ': gt_occ[0:ix],
            'pts': gt_pts[0:ix],
            'distance':gt_dist[0:ix],
            'difficulty': gt_diff[0:ix],
            'alpha': gt_alpha[0:ix],
            'ids': gt_ids[0:ix],
            'cat': cat,
            'boxes': boxes[0:ix],
            'boxes_dc': boxes_dc[0:ix_dc],
            'gt_classes': gt_classes[0:ix],
            'gt_overlaps': overlaps[0:ix],
            'flipped': False,
            'seg_areas': seg_areas[0:ix]
        }

    def _load_scene_meta(self, scene_desc_filepath):
        self._scene_meta = {}
        with open(scene_desc_filepath, newline='\n') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for i, row in enumerate(reader):
                for j, elem in enumerate(row):
                    if(i == 0):
                        elem_t = elem.replace(' ','_').lower()
                        self._scene_meta[elem_t] = []
                    else:
                        for k, (key,_) in enumerate(self._scene_meta.items()):
                            if j == k:
                                self._scene_meta[key].append(elem)
        self._scene_meta_postprocess()
        return self._scene_meta

    def _scene_meta_postprocess(self):
        drive  = self._scene_meta['date']
        seq    = self._scene_meta['number']
        snow   = self._scene_meta['snow_points_removed']
        self._scene_meta['cam00_obs'] = self._scene_meta['cam_00_lens_snow_cover']
        self._scene_meta['scene_idx'] = []
        self._scene_meta['scene_desc'] = []
        self._scene_meta['cam_obs'] = []
        for i in range(0,len(drive)):
            scene_idx = int(re.sub('[^0-9]','',drive[i]))*100 + int(seq[i])
            self._scene_meta['scene_idx'].append(scene_idx)
            snow_tmp = int(snow[i])
            if(snow_tmp < 25):
                scene_desc = 'none'
            elif(snow_tmp < 250):
                scene_desc = 'light'
            elif(snow_tmp < 500):
                scene_desc = 'medium'
            elif(snow_tmp < 750):
                scene_desc = 'heavy'
            else:
                scene_desc = 'extreme'
            self._scene_meta['scene_desc'].append(scene_desc)
            
    def _get_scene_desc(self, scene_idx):
        all_scene_idx = self._scene_meta['scene_idx']
        loc = all_scene_idx.index(scene_idx)
        all_scene_desc = self._scene_meta['scene_desc']
        return all_scene_desc[loc]

    def draw_and_save_eval(self,filename,roi_dets,roi_det_labels,dets,uncertainties,iter,mode,draw_folder=None,frame_arr=None):
        out_dir = self._find_draw_folder(mode, draw_folder)
        if(iter != 0):
            out_file = 'iter_{}_'.format(iter) + os.path.basename(filename).replace('.{}'.format(self._filetype.lower()),'.{}'.format(self._imtype.lower()))
        else:
            out_file = 'img-'.format(iter) + os.path.basename(filename).replace('.{}'.format(self._filetype.lower()),'.{}'.format(self._imtype.lower()))
        out_file = os.path.join(out_dir,out_file)
        if(frame_arr is None):
            source_img = Image.open(filename)
        else:
            img_arr = frame_arr[0]
            img_arr = img_arr*cfg.PIXEL_STDDEVS
            img_arr += cfg.PIXEL_MEANS
            img_arr = img_arr[:,:,cfg.PIXEL_ARRANGE_BGR]
            img_arr = img_arr.astype(np.uint8)
            source_img = Image.fromarray(img_arr)
        draw = ImageDraw.Draw(source_img)
        for class_dets in dets:
            #Set of detections, one for each class
            for det in class_dets:
                draw.rectangle([(det[0],det[1]),(det[2],det[3])],outline=(0,int(det[4]*255),0))
        for det,label in zip(roi_dets,roi_det_labels):
            if(label == 1):
                color = (255,255,255)
            else:
                color = (0,0,0)
            if(label == 1):
                draw.rectangle([(det[0],det[1]),(det[2],det[3])],outline=color)
        print('Saving file at location {}'.format(out_file))
        source_img.save(out_file,self._imtype)

    def _do_python_eval(self, output_dir='output', mode='train'):
        frame_index = self._get_index_for_mode(mode)
        #annopath is for labels and only labelled images
        annopath = os.path.join(self._devkit_path, mode, 'label_2')
        print(annopath)
        num_d_levels = 1
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
            #cadc/results/comp_X_testing_class.txt
            detfile = self._get_results_file_template(mode,cls,output_dir)
            #Run cadc evaluation metrics on each image
            rec, prec, ap = cadc_eval(
                detfile,
                self,
                frame_index,
                cls,
                self._get_cache_dir(),
                mode,
                ovthresh=ovt,
                eval_type='2d',
                d_levels=num_d_levels)
            aps[i-1,:] = ap
            #Tell user of AP
            print_str = 'AP for {} = '.format(cls)
            for d_lvl in range(num_d_levels):
                print_str += '{}: {:.4f} '.format(d_lvl,np.mean(aps[:,d_lvl]))
            print(print_str)
            #print(('AP for {} = E {:.4f} M {:.4f} H {:.4f}'.format(cls,ap[0],ap[1],ap[2])))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print_str = 'Mean AP = '
        for d_lvl in range(num_d_levels):
            print_str += '{}: {:.4f} '.format(d_lvl,np.mean(aps[:,d_lvl]))
        print(print_str)
        #print(('Mean AP = E {:.4f} M {:.4f} H {:.4f}'.format(np.mean(aps[:,0]),np.mean(aps[:,1]),np.mean(aps[:,2]))))

    def _do_matlab_eval(self, output_dir='output'):
        print('-----------------------------------------------------')
        print('Computing results with the official MATLAB eval code.')
        print('-----------------------------------------------------')
        path = os.path.join(cfg.ROOT_DIR, 'lib', 'datasets',
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'cadc_eval(\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
            .format(self._devkit_path,  self._mode, output_dir)
        print(('Running:\n{}'.format(cmd)))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir, mode):
        print('writing results to file...')
        self._write_image_results_file(all_boxes, output_dir, mode)
        self._do_python_eval(output_dir, mode)
        if self.config['matlab_eval']:
            self._do_matlab_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == 'dontcare'  or cls == '__background__':
                    continue
                filename = self._get_results_file_template(mode,cls,output_dir)
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
