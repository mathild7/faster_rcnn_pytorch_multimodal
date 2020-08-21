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
# import scipy.io as sio
import pickle
from PIL import Image, ImageDraw
import subprocess
import uuid
import traceback
from .kitti_eval import kitti_eval
from model.config import cfg, get_output_dir
import shutil
import utils.kitti_utils as kitti_utils
from utils.kitti_utils import Calibration as kitti_calib
from random import SystemRandom
import utils.bbox as bbox_utils

class kitti_lidb(db):
    def __init__(self, mode='test',limiter=0, shuffle_en=True):
        name = 'kitti'
        db.__init__(self, name, mode)
        self._devkit_path = self._get_default_path()
        self._data_path = self._devkit_path
        self._mode = mode
        self._uncertainty_sort_type = cfg.UC.SORT_TYPE
        self._draw_width = int((cfg.LIDAR.X_RANGE[1] - cfg.LIDAR.X_RANGE[0])*(1/cfg.LIDAR.VOXEL_LEN))
        self._draw_height = int((cfg.LIDAR.Y_RANGE[1] - cfg.LIDAR.Y_RANGE[0])*(1/cfg.LIDAR.VOXEL_LEN))
        self._num_slices = cfg.LIDAR.NUM_SLICES
        self._frame_sub_dir = 'velodyne'
        self._train_dir = os.path.join(self._data_path, 'training', self._frame_sub_dir)
        self._val_dir   = os.path.join(self._data_path, 'training', self._frame_sub_dir)
        self._test_dir   = os.path.join(self._data_path, 'testing', self._frame_sub_dir)
        self._split_dir  = os.path.join(self._data_path, 'splits')
        self._test_index = open(self._split_dir+'/test.txt').read().splitlines()
        self._train_index = open(self._split_dir+'/train.txt').read().splitlines()
        self._val_index = open(self._split_dir+'/val.txt').read().splitlines()
        self._filetype   = 'bin'
        self._imtype   = 'PNG'
        self.type = 'lidar'
        self._bev_slice_locations = [1,2,3,4,5,7]
        self._mode = mode
        #Backwards compatibility
        self._train_sub_folder = 'training'
        self._val_sub_folder = 'training'
        self._test_sub_folder = 'testing'
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
        #self._train_index = sorted([d for d in os.listdir(self._train_dir) if d.endswith('.bin')])
        #self._val_index   = sorted([d for d in os.listdir(self._val_dir) if d.endswith('.bin')])
        #self._test_index  = sorted([d for d in os.listdir(self._test_dir) if d.endswith('.bin')])
        #Limiter
        if(limiter != 0):
            if(limiter < len(self._val_index)):
                self._val_index   = self._val_index[:limiter]
            if(limiter < len(self._train_index)):
                self._train_index = self._train_index[:limiter]
            if(limiter < len(self._test_index)):
                self._test_index = self._test_index[:limiter]
        rand = SystemRandom()
        if(shuffle_en):
            print('shuffling frame indices')
            rand.shuffle(self._val_index)
            rand.shuffle(self._train_index)
            rand.shuffle(self._test_index)
        assert os.path.exists(self._devkit_path), \
            'Kitti dataset path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)

    def subfolder_from_mode(self, mode):
        if(mode == 'train'):
            return self._train_sub_folder
        if(mode == 'val'):
            return self._val_sub_folder
        if(mode == 'test'):
            return self._test_sub_folder
        return None

    def path_from_index(self, mode, index):
        """
    Construct an image path from the image's "index" identifier.
    """
        mode_sub_folder = self.subfolder_from_mode(mode)
        image_path = os.path.join(self._devkit_path, mode_sub_folder, 'velodyne', index+'.bin')
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def _get_default_path(self):
        """
    Return the default path where PASCAL VOC is expected to be installed.
    """
        return os.path.join(cfg.DATA_DIR, 'kitti')

    def gt_roidb(self, mode):
        """
    Return the database of ground-truth regions of interest.

    This function loads/saves from/to a cache file to speed up future calls.
    """
        #for line in traceback.format_stack():
        #    print(line.strip())
        cache_dir = self._get_cache_dir()
        cache_file = os.path.join(cache_dir, self._name + '_' + mode + '_lidar_gt_roidb.pkl')
        frame_index = self._get_index_for_mode(mode)
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                try:
                    roidb = pickle.load(fid)
                except:
                    roidb = pickle.load(fid, encoding='bytes')
            print('{} gt roidb loaded from {}'.format(self._name, cache_file))
            return roidb

        gt_roidb = []
        for frame in frame_index:
            idx = frame.replace('.{}'.format(self._filetype),'')
            roi = self._load_kitti_annotation(idx, mode)
            if(roi is not None):
                gt_roidb.append(roi)
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    #Only care about foreground classes
    def _load_kitti_annotation(self, index, mode='train', remove_without_gt=True):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        #print('loading kitti anno')
        filename = os.path.join(self._data_path, self.mode_to_sub_folder(mode), 'label_2', index + '.txt')
        calib_file = os.path.join(self._data_path, self.mode_to_sub_folder(mode), 'calib', index + '.txt')
        img_filename = os.path.join(self._data_path, self.mode_to_sub_folder(mode), 'velodyne', index + '.{}'.format(self._filetype))
        label_lines = open(filename, 'r').readlines()
        num_objs = len(label_lines)
        boxes      = np.zeros((num_objs, cfg.LIDAR.NUM_BBOX_ELEM), dtype=np.float32)
        boxes_dc   = np.zeros((num_objs, cfg.LIDAR.NUM_BBOX_ELEM), dtype=np.float32)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        ignore     = np.zeros((num_objs), dtype=np.bool)
        gt_diff    = np.zeros((num_objs), dtype=np.int8)
        gt_trunc   = np.zeros((num_objs), dtype=np.float32)
        gt_occ     = np.zeros((num_objs), dtype=np.int16)
        gt_ids     = np.zeros((num_objs), dtype=np.int16)
        gt_dist    = np.zeros((num_objs), dtype=np.float32)
        cat = []
        overlaps   = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas  = np.zeros((num_objs), dtype=np.float32)
        calib = kitti_calib(calib_file)
        # Load object bounding boxes into a data frame.
        ix = 0
        ix_dc = 0
        populated_idx = []
        for line in label_lines:
            label_arr = line.split(' ')
            # Make pixel indexes 0-based
            trunc = float(label_arr[1])
            occ   = int(label_arr[2])
            x1 = float(label_arr[4])
            y1 = float(label_arr[5])
            x2 = float(label_arr[6])
            y2 = float(label_arr[7])
            BBGT_height = y2 - y1
            h_z = float(label_arr[8])
            w_y = float(label_arr[9])
            l_x = float(label_arr[10])
            x_c = float(label_arr[11])
            y_c = float(label_arr[12])
            z_c = float(label_arr[13])
            heading = float(label_arr[14].replace('\n',''))
            #Lock headings to be [pi/2, -pi/2)
            pi2 = float(np.pi/2.0)
            heading = -heading + pi2
            #if(heading > pi2):
            #    heading = heading - np.pi
            #if(heading <= -pi2):
            #    heading = heading + np.pi
            bbox = [x_c, y_c, z_c, l_x, w_y, h_z, heading]
            bbox = self._bbox3d(bbox, calib)
            #Translate to PC reference frame
            trunc = float(label_arr[1])
            occ   = int(label_arr[2])
            if(occ <= 0 and trunc <= 0.15 and (BBGT_height) >= 40):
                diff = 0
            elif(occ <= 1 and trunc <= 0.3 and (BBGT_height) >= 25):
                diff = 1
            elif(occ <= 2 and trunc <= 0.5 and (BBGT_height) >= 25):
                diff = 2
            else:
                diff = 3
            #If car doesn't fit inside 2 voxels minimum
            if(bbox[1] - bbox[4]/2 >= cfg.LIDAR.Y_RANGE[1] - cfg.LIDAR.VOXEL_LEN*2):
                continue
            if(bbox[0] - bbox[3]/2 >= cfg.LIDAR.X_RANGE[1] - cfg.LIDAR.VOXEL_LEN*2):
                continue
            if(bbox[1] + bbox[4]/2 < cfg.LIDAR.Y_RANGE[0] + cfg.LIDAR.VOXEL_LEN*2):
                continue
            if(bbox[0] + bbox[3]/2 < cfg.LIDAR.X_RANGE[0] + cfg.LIDAR.VOXEL_LEN*2):
                continue
            if(label_arr[0].strip() not in self._classes):
                #print('replacing {:s} with dont care'.format(label_arr[0]))
                label_arr[0] = 'dontcare'
            if('dontcare' not in label_arr[0].lower().strip()):
                #print(label_arr)
                cls = self._class_to_ind[label_arr[0].strip()]
                cat.append(label_arr[0].strip())
                boxes[ix, :] = bbox
                gt_classes[ix] = cls
                gt_trunc[ix] = trunc
                gt_occ[ix]   = occ
                gt_dist[ix]  = x_c
                gt_diff[ix]  = diff
                gt_ids[ix]   = int(index)*100 + ix
                #overlaps is (NxM) where N = number of GT entires and M = number of classes
                overlaps[ix, cls] = 1.0
                seg_areas[ix] = 0
                ix = ix + 1
            if('dontcare' in label_arr[0].lower().strip()):
                #print(line)
                boxes_dc[ix_dc, :] = np.zeros((cfg.LIDAR.NUM_BBOX_ELEM))
                ix_dc = ix_dc + 1
                

        overlaps = scipy.sparse.csr_matrix(overlaps)
        #assert(len(boxes) != 0, "Boxes is empty for label {:s}".format(index))
        if(ix == 0 and remove_without_gt is True):
            print('removing pc {} with no GT boxes specified'.format(index))
            return None
        return {
            'idx':          index,
            'ids':          gt_ids[0:ix],
            'filename':     img_filename,
            'det':          ignore[0:ix].copy(),
            'ignore':       ignore[0:ix],
            'hit':          ignore[0:ix].copy(),
            'trunc':        gt_trunc[0:ix],
            'occ':          gt_occ[0:ix],
            'difficulty':   gt_diff[0:ix],
            'cat':          cat,
            'boxes':        boxes[0:ix],
            'boxes_dc':     boxes_dc[0:ix_dc],
            'gt_classes':   gt_classes[0:ix],
            'gt_overlaps':  overlaps[0:ix],
            'distance':     gt_dist[0:ix],
            'flipped':      False,
            'seg_areas':    seg_areas[0:ix]
        }

    def _bbox3d(self, bbox, calib):
        #box3d_pts_2d, box3d_pts_3d = kitti_utils.compute_box_3d(bbox, calib.P)
        xyz = [[bbox[0],bbox[1],bbox[2]]]
        xyz = calib.project_rect_to_velo(xyz)
        bbox[0] = xyz[0][0]
        bbox[1] = xyz[0][1]
        bbox[2] = xyz[0][2]
        return bbox

    #DEPRECATED
    #def draw_and_save(self,mode,image_token=None):
    #    datapath = os.path.join(cfg.DATA_DIR, self._name)
    #    out_file = os.path.join(cfg.DATA_DIR, self._name, self.mode_to_sub_folder(mode),'drawn')
    #    print('deleting files in dir {}'.format(out_file))
    #    if(os.path.isdir(datapath)):
    #        shutil.rmtree(datapath)
    #    os.makedirs(out_file)
    #    if(mode == 'val'):
    #        roidb = self.val_roidb
    #    elif(mode == 'train'):
    #        roidb = self.roidb
    #    #print('about to draw in {} mode with ROIDB size of {}'.format(mode,len(roidb)))
    #    for i, roi in enumerate(roidb):
    #        if(i % 250 == 0):
    #            if(roi['flipped']):
    #                outfile = roi['filename'].replace('/image_2','/drawn').replace('.{}'.format(self._imtype.lower()),'_flipped.{}'.format(self._imtype.lower()))
    #            else:
    #                outfile = roi['filename'].replace('/image_2','/drawn')
    #            if(roi['boxes'].shape[0] != 0):
    #                source_img = Image.open(roi['filename'])
    #                if(roi['flipped'] is True):
    #                    source_img = source_img.transpose(Image.FLIP_LEFT_RIGHT)
    #                    text = "Flipped"
    #                else:
    #                    text = "Normal"
    #                draw = ImageDraw.Draw(source_img)
    #                draw.text((0,0),text)
    #                for roi_box,cat in zip(roi['boxes'],roi['cat']):
    #                    draw.text((roi_box[0],roi_box[1]),cat)
    #                    draw.rectangle([(roi_box[0],roi_box[1]),(roi_box[2],roi_box[3])],outline=(0,255,0))
    #                for roi_box in roi['boxes_dc']:
    #                    draw.rectangle([(roi_box[0],roi_box[1]),(roi_box[2],roi_box[3])],outline=(255,0,0))
    #                print('Saving drawn file at location {}'.format(outfile))
    #                source_img.save(outfile,self._imtype)

    def _load_pc(self,filename):
        return np.fromfile(filename, dtype=np.float32).reshape(-1, 4)

    def _draw_bev_pseudo_img(self,voxel_grid):
        voxel_grid_rgb = np.zeros((voxel_grid.shape[0],voxel_grid.shape[1],3))
        voxel_grid_rgb[:,:,0] = np.max(voxel_grid[:,:,0:cfg.LIDAR.NUM_SLICES],axis=2)
        max_height = np.max(voxel_grid_rgb[:,:,0])
        min_height = np.min(voxel_grid_rgb[:,:,0])
        voxel_grid_rgb[:,:,0] = np.clip(voxel_grid_rgb[:,:,0]*(255/(max_height - min_height)),0,255)
        voxel_grid_rgb[:,:,1] = voxel_grid[:,:,cfg.LIDAR.NUM_SLICES]*(255/voxel_grid[:,:,cfg.LIDAR.NUM_SLICES].max())
        voxel_grid_rgb[:,:,2] = voxel_grid[:,:,cfg.LIDAR.NUM_SLICES+1]*(255/voxel_grid[:,:,cfg.LIDAR.NUM_SLICES+1].max())
        voxel_grid_rgb        = voxel_grid_rgb.astype(dtype='uint8')
        draw_file = Image.fromarray(voxel_grid_rgb,'RGB')
        return draw_file

    #TODO: Merge with waymo lidb draw and save eval, image specific
    def draw_and_save_eval(self,filename,roi_dets,roi_det_labels,dets,uncertainties,iter,mode,draw_folder=None,frame_arr=None):
        out_dir = self._find_draw_folder(mode, draw_folder)
        out_file = 'iter_{}_'.format(iter) + os.path.basename(filename).replace('.{}'.format(self._filetype.lower()),'.{}'.format(self._imtype.lower()))
        out_file = os.path.join(out_dir,out_file)
        #out_file = filename.replace('/point_clouds/','/{}_drawn/iter_{}_'.format(mode,iter)).replace('.{}'.format(self._filetype.lower()),'.{}'.format(self._imtype.lower()))
        if(frame_arr is None):
            source_bin = self._load_pc(filename)
            draw_file  = Image.new('RGB', (self._draw_width,self._draw_height), (0,0,0))
            draw = ImageDraw.Draw(draw_file)
            self.draw_bev(source_bin,draw)
        else:
            draw_file = self._draw_bev_pseudo_img(frame_arr[0])
            draw = ImageDraw.Draw(draw_file)
        #TODO: Magic numbers
        limiter = 10
        y_start = self._draw_height - 10*(limiter+2)
        #TODO: Swap axes of dets
        if(len(roi_dets) > 0):
            if(roi_dets.shape[1] == 4):
                for det,label in zip(roi_dets,roi_det_labels):
                    if(label == 0):
                        color = 127
                    else:
                        color = 255
                    draw.rectangle([(det[0],det[1]),(det[2],det[3])],outline=(color,color,color))
            elif(roi_dets.shape[1] == 7):
                for det,label in zip(roi_dets,roi_det_labels):
                    if(label == 0):
                        colors = [127,127,127]
                    else:
                        colors = [255,255,255]
                    bbox_utils.draw_bev_bbox(draw,det,[self._draw_width, self._draw_height, cfg.LIDAR.Z_RANGE[1]-cfg.LIDAR.Z_RANGE[0]],transform=False, colors=colors)

        for j,class_dets in enumerate(dets):
            #Set of detections, one for each class
            #Ignore background
            if(j > 0):
                if(len(class_dets) > 0):
                    cls_uncertainties = self._normalize_uncertainties(class_dets,uncertainties[j])
                    #cls_uncertainties = uncertainties[j]
                    det_idx = self._sort_dets_by_uncertainty(class_dets,cls_uncertainties,descending=True)
                    avg_det_string = 'avg: '
                    num_det = len(det_idx)
                    if(num_det < limiter):
                        limiter = num_det
                    for i,idx in enumerate(det_idx):
                        uc_gradient = int((limiter-i)/limiter*255.0)
                        det = class_dets[idx]
                        #print(det)
                        if(det.shape[0] > 5):
                            colors = [0,int(det[7]*255),0]
                            bbox_utils.draw_bev_bbox(draw,det,[self._draw_width, self._draw_height, cfg.LIDAR.Z_RANGE[1]-cfg.LIDAR.Z_RANGE[0]],transform=False, colors=colors)
                        else:
                            color_g = int(det[4]*255)
                            color_b = int(1-det[4])*255
                            draw.rectangle([(det[0],det[1]),(det[2],det[3])],outline=(0,color_g,color_b))
                        det_string = '{:02} '.format(i)
                        if(i < limiter):
                            draw.text((det[0]+4,det[1]+4),det_string,fill=(0,int(det[-1]*255),uc_gradient,255))
                        for key,val in cls_uncertainties.items():
                            if('cls' in key):
                                key = key.replace('cls','c').replace('bbox','b').replace('mutual_info','m_i')
                                if(i == 0):
                                    avg_det_string += '{}: {:5.3f} '.format(key,np.mean(np.mean(val)))
                                det_string += '{}: {:5.3f} '.format(key,np.mean(val[idx]))
                            else:
                                if(i == 0):
                                    avg_det_string += '{}: {:5.3f} '.format(key,np.mean(np.mean(val)))
                                det_string += '{}: {:5.3f} '.format(key,np.mean(val[idx]))
                        det_string += 'con: {:5.3f} '.format(det[-1])
                        if(i < limiter):
                            draw.text((0,y_start+i*10),det_string, fill=(0,int(det[4]*255),uc_gradient,255))
                    draw.text((0,self._draw_height-10),avg_det_string, fill=(255,255,255,255))
                elif(cfg.DEBUG.EN_TEST_MSG):
                    print('draw and save: No detections for pc {}, class: {}'.format(filename,j))
        print('Saving BEV map file at location {}'.format(out_file))
        draw_file.save(out_file,self._imtype)

    def _do_python_eval(self, output_dir='output', mode='train'):
        frame_index = self._get_index_for_mode(mode)
        #annopath is for labels and only labelled images
        annopath = os.path.join(self._devkit_path, self.mode_to_sub_folder(mode), 'label_2')
        print(annopath)
        num_d_levels = 3
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
            detfile = self._get_results_file_template(mode,cls,output_dir)
            #Run kitti evaluation metrics on each image
            rec, prec, ap = kitti_eval(
                detfile,
                self,
                frame_index,
                cls,
                cachedir,
                mode,
                ovthresh=ovt,
                eval_type='bev',
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
        cmd += 'kitti_eval(\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
            .format(self._devkit_path,  self._mode, output_dir)
        print(('Running:\n{}'.format(cmd)))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir, mode):
        print('writing results to file...')
        self._write_lidar_results_file(all_boxes, output_dir, mode)
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
