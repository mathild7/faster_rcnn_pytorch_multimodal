# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import shutil
import os
import json
from datasets.db import db
# import datasets.ds_utils as ds_utils
#import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
# import scipy.io as sio
from enum import Enum
import pickle
from PIL import Image, ImageDraw
from random import SystemRandom
from shapely.geometry import MultiPoint, box
#Useful for debugging without a IDE
#import traceback
from .waymo_eval import waymo_eval
from model.config import cfg, get_output_dir
import utils.bbox as bbox_utils
import re

def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)',text) ]

class class_enum(Enum):
    UNKNOWN = 0
    VEHICLE = 1
    PEDESTRIAN = 2
    SIGN = 3
    CYCLIST = 4

class waymo_lidb(db):
    def __init__(self, mode='test',limiter=0, shuffle_en=False):
        name = 'waymo'
        self.type = 'lidar'
        db.__init__(self, name, mode)
        self._train_scenes = []
        self._val_scenes = []
        self._test_scenes = []
        if(mode == 'test'):
            self._tod_filter_list = cfg.TEST.TOD_FILTER_LIST
        else:
            self._tod_filter_list = cfg.TRAIN.TOD_FILTER_LIST
        self._uncertainty_sort_type = cfg.UC.SORT_TYPE
        self._draw_width = int((cfg.LIDAR.X_RANGE[1] - cfg.LIDAR.X_RANGE[0])*(1/cfg.LIDAR.VOXEL_LEN))
        self._draw_height = int((cfg.LIDAR.Y_RANGE[1] - cfg.LIDAR.Y_RANGE[0])*(1/cfg.LIDAR.VOXEL_LEN))
        self._num_slices = cfg.LIDAR.NUM_SLICES
        self._bev_slice_locations = [1,2,3,4,5,7]
        self._filetype   = 'npy'
        self._imtype   = 'PNG'
        self._scene_sel = True
        #For now one large cache file is OK, but ideally just take subset of actually needed data and cache that. No need to load nusc every time.

        self._classes = (
            'dontcare',  # always index 0
            'vehicle.car')
           # 'human.pedestrian',
            #'vehicle.bicycle')
        self.config = {
            'cleanup': True,
            'matlab_eval': False,
            'rpn_file': None
        }
        self._class_to_ind = dict(
            list(zip(self.classes, list(range(self.num_classes)))))

        self._train_index = os.listdir(os.path.join(self._devkit_path,'train','point_clouds'))
        self._val_index   = os.listdir(os.path.join(self._devkit_path,'val','point_clouds'))
        self._val_index.sort(key=natural_keys)
        rand = SystemRandom()
        if(shuffle_en):
            print('shuffling pc indices')
            rand.shuffle(self._train_index)
            rand.shuffle(self._val_index)
        if(limiter != 0):
            if(limiter < len(self._val_index)):
                self._val_index   = self._val_index[:limiter]
            if(limiter < len(self._train_index)):
                self._train_index = self._train_index[:limiter]
            if(limiter < len(self._test_index)):
                self._test_index = self._test_index[:limiter]
        assert os.path.exists(self._devkit_path), 'waymo dataset path does not exist: {}'.format(self._devkit_path)

    def _load_pc(self,filename):
        return np.load(filename)

    def path_from_index(self, mode, index):
        """
    Construct an pc path from the pc's "index" identifier.
    """
        pc_path = os.path.join(self._devkit_path, mode, 'point_clouds', index)
        assert os.path.exists(pc_path), \
            'Path does not exist: {}'.format(pc_path)
        return pc_path

    def _get_default_path(self):
        """
    Return the default path where PASCAL VOC is expected to be installed.
    """
        return os.path.join(cfg.DATA_DIR, 'waymo')

    def gt_roidb(self,mode='train'):
        """
    Return the database of ground-truth regions of interest.

    This function loads/saves from/to a cache file to speed up future calls.
    """
        cache_file = os.path.join(self._devkit_path, 'cache', self._name + '_' + mode + '_lidar_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                try:
                    roidb = pickle.load(fid)
                except:
                    roidb = pickle.load(fid, encoding='bytes')
            print('{} gt roidb loaded from {}'.format(self._name, cache_file))
            return roidb
        labels_filename = os.path.join(self._devkit_path, mode,'labels/lidar_labels.json')
        gt_roidb = []
        with open(labels_filename,'r') as labels_file:
            data = labels_file.read()
            #print(data)
            #print(data)
            labels = json.loads(data)
            index = None
            sub_total   = 0
            if(mode == 'train'):
                index = self._train_index
            elif(mode == 'val'):
                index = self._val_index
            for pc in index:
                #print(pc)
                for pc_labels in labels:
                    #print(pc_labels['assoc_frame'])
                    if(pc_labels['assoc_frame'] == pc.replace('.{}'.format(self._filetype.lower()),'')):
                        pc_file_path = os.path.join(mode,'point_clouds',pc)
                        roi = self._load_waymo_annotation(pc_file_path,pc_labels,tod_filter_list=self._tod_filter_list)
                        if(roi is None):
                            sub_total += 1
                        else:
                            gt_roidb.append(roi)
                        break
            with open(cache_file, 'wb') as fid:
                pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
            print('wrote gt roidb to {}'.format(cache_file))
        return gt_roidb

    #DEPRECATED
    #def draw_and_save(self,mode,pc_token=None):
    #    datapath = os.path.join(cfg.DATA_DIR, self._name)
    #    out_file = os.path.join(cfg.DATA_DIR, self._name, mode,'drawn')
    #    print('deleting files in dir {}'.format(out_file))
    #    if(os.path.isdir(out_file)):
    #        shutil.rmtree(out_file)
    #    os.makedirs(out_file)
    #    if(mode == 'val'):
    #        roidb = self.val_roidb
    #    elif(mode == 'train'):
    #        roidb = self.roidb
    #    #print('about to draw in {} mode with ROIDB size of {}'.format(mode,len(roidb)))
    #    for i, roi in enumerate(roidb):
    #        if(i % 1 == 0):
    #            if(roi['boxes'].shape[0] != 0):
    #                source_bin = np.fromfile(roi['pcfile'], dtype='float32').reshape((-1,5))
    #                bev_img    = bbox_utils.point_cloud_to_bev(source_bin)
    #                outfile = roi['pcfile'].replace('/point_clouds','/drawn').replace('.{}'.format(self._filetype.lower()),'.{}'.format(self._imtype.lower()))
    #                draw_file  = Image.new('RGB', (self._draw_width,self._draw_height), (255,255,255))
    #                draw = ImageDraw.Draw(draw_file)
    #                self.draw_bev(source_bin,draw)
    #                for roi_box, cat in zip(roi['boxes'],roi['cat']):
    #                    bbox_utils.draw_bev_bbox(draw,roi_box,None)
    #                #print('Saving entire BEV drawn file at location {}'.format(outfile))
    #                draw_file.save(outfile,self._imtype)
    #                #for slice_idx, bev_slice in enumerate(bev_img):
    #                #    outfile = roi['pcfile'].replace('/point_clouds','/drawn').replace('.{}'.format(self._filetype.lower()),'_{}.{}'.format(slice_idx,self._imtype.lower()))
    #                #    draw_file  = Image.new('RGB', (self._draw_width,self._draw_height), (255,255,255))
    #                #    draw = ImageDraw.Draw(draw_file)
    #                #    if(roi['flipped'] is True):
    #                #        #source_img = source_bin.transpose(Image.FLIP_LEFT_RIGHT)
    #                #        text = "Flipped"
    #                #    else:
    #                #        text = "Normal"
    #                #    draw = self.draw_bev_slice(bev_slice,slice_idx,draw)
    #                #    draw.text((0,0),text)
    #                #    for roi_box, cat in zip(roi['boxes'],roi['cat']):
    #                #        bbox_utils.draw_bev_bbox(draw,roi_box,slice_idx)
    #                #        #draw.text((roi_box[0],roi_box[1]),cat)
    #                #    #for roi_box in roi['boxes_dc']:
    #                #    #    draw.rectangle([(roi_box[0],roi_box[1]),(roi_box[2],roi_box[3])],outline=(255,0,0))
    #                #    print('Saving BEV slice {} drawn file at location {}'.format(slice_idx,outfile))
    #                #    draw_file.save(outfile,self._imtype)

    #DEPRECATED
    #def point_cloud_to_bev(self,pc):
    #    pc_slices = []
    #    for i in range(0,self._num_slices):
    #        z_max, z_min = self._slice_height(i)
    #        pc_min_thresh = pc[(pc[:,2] >= z_min)]
    #        pc_min_and_max_thresh = pc_min_thresh[(pc_min_thresh[:,2] < z_max)]
    #        pc_slices.append(pc_min_and_max_thresh)
    #    bev_img = np.array(pc_slices)
    #    return bev_img

    def _slice_height(self,i):
        if(i == 0):
            z_min = cfg.LIDAR.Z_RANGE[0]
            z_max = self._bev_slice_locations[i]
        elif(i == self._num_slices-1):
            z_min = self._bev_slice_locations[i-1]
            z_max = cfg.LIDAR.Z_RANGE[1]
        else:
            z_min = self._bev_slice_locations[i-1]
            z_max = self._bev_slice_locations[i]
        return z_max, z_min

    def draw_and_save_eval(self,filename,roi_dets,roi_det_labels,dets,uncertainties,iter,mode,draw_folder=None):
        out_dir = self._find_draw_folder(mode, draw_folder)
        out_file = 'iter_{}_'.format(iter) + os.path.basename(filename).replace('.{}'.format(self._filetype.lower()),'.{}'.format(self._imtype.lower()))
        out_file = os.path.join(out_dir,out_file)
        #out_file = filename.replace('/point_clouds/','/{}_drawn/iter_{}_'.format(mode,iter)).replace('.{}'.format(self._filetype.lower()),'.{}'.format(self._imtype.lower()))
        source_bin = self._load_pc(filename)
        draw_file  = Image.new('RGB', (self._draw_width,self._draw_height), (0,0,0))
        draw = ImageDraw.Draw(draw_file)
        self.draw_bev(source_bin,draw)
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

    #Only care about foreground classes
    def _load_waymo_annotation(self, pc_file_path, pc_labels, remove_without_gt=True,tod_filter_list=[],filter_boxes=False):
        filename = os.path.join(self._devkit_path, pc_file_path)
        num_objs = len(pc_labels['box'])

        boxes      = np.zeros((num_objs, cfg.LIDAR.NUM_BBOX_ELEM), dtype=np.float32)
        boxes_dc   = np.zeros((num_objs, cfg.LIDAR.NUM_BBOX_ELEM), dtype=np.float32)
        ids        = []
        num_pts    = np.zeros((num_objs,), dtype=np.int32)
        cat        = []
        difficulty = np.zeros((num_objs, ),dtype=np.int32)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        ignore     = np.zeros((num_objs), dtype=np.bool)
        overlaps   = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        weather = pc_labels['scene_type'][0]['weather']
        tod = pc_labels['scene_type'][0]['tod']
        pc_id = pc_labels['id']
        scene_desc = json.dumps(pc_labels['scene_type'][0])
        #TODO: Magic number
        scene_idx  = int(int(pc_labels['assoc_frame']) / cfg.MAX_IMG_PER_SCENE)
        pc_idx    = int(int(pc_labels['assoc_frame']) % cfg.MAX_IMG_PER_SCENE)
        #Removing night-time/day-time ROI's
        if(tod not in tod_filter_list):
            print('TOD {} not in specified filter list'.format(tod))
            return None
        #seg_areas  = np.zeros((num_objs), dtype=np.float32)
        extrinsic = pc_labels['calibration'][0]['extrinsic_transform']
        #camera_intrinsic = pc_labels['calibration'][0]['intrinsic']
        # Load object bounding boxes into a data frame.
        ix = 0
        ix_dc = 0
        for i, bbox in enumerate(pc_labels['box']):
            diff       = pc_labels['difficulty'][i]
            anno_cat   = pc_labels['class'][i]
            track_id   = pc_labels['id'][i]
            pts        = pc_labels['meta'][i]['pts']
            if(class_enum(anno_cat) == class_enum.SIGN):
                anno_cat = class_enum.UNKNOWN.value
            elif(class_enum(anno_cat) == class_enum.CYCLIST):
                #Sign is taking index 3, where my code expects cyclist to be. Therefore replace any cyclist (class index 4) with sign (class index 3)
                anno_cat = class_enum.SIGN.value

            #OVERRIDE
            if(class_enum(anno_cat) != class_enum.VEHICLE):
                anno_cat = class_enum.UNKNOWN.value
            #Change to string 
            anno_cat = self._classes[anno_cat]
            x_c = float(bbox['xc'])
            y_c = float(bbox['yc'])
            z_c = float(bbox['zc'])
            l_x = float(bbox['lx'])
            w_y = float(bbox['wy'])
            h_z = float(bbox['hz'])
            heading = float(bbox['heading'])
            #Lock headings to be [pi/2, -pi/2)
            pi2 = float(np.pi/2.0)
            heading = np.where(heading > pi2, heading - np.pi, heading)
            heading = np.where(heading <= -pi2, heading + np.pi, heading)
            bbox = [x_c, y_c, z_c, l_x, w_y, h_z, heading]
            #Clip bboxes eror checking
            #Pointcloud to be cropped at x=[-40,40] y=[0,70] z=[0,10]
            #if(x1 < cfg.LIDAR.X_RANGE[0] or x2 > cfg.LIDAR.X_RANGE[1]):
            #    print('x1: {} x2: {}'.format(x1,x2))
            #if(y1 < cfg.LIDAR.Y_RANGE[0] or y2 > cfg.LIDAR.Y_RANGE[1]):
            #    print('y1: {} y2: {}'.format(y1,y2))
            #if(z1 < cfg.LIDAR.Z_RANGE[0] or z2 > cfg.LIDAR.Z_RANGE[1]):
            #    print('z1: {} z2: {}'.format(z1,z2))

            if(anno_cat != 'dontcare'):
                #print(label_arr)
                cls = self._class_to_ind[anno_cat]
                #Stop little clips from happening for cars
                boxes[ix, :]   = bbox
                difficulty[ix] = diff
                num_pts[ix]    = pts
                ids.append(track_id)
                #TODO: Not sure what to filter these to yet.
                #if(anno_cat == 'vehicle.car' and self._mode == 'train'):
                    #TODO: Magic Numbers
                    #if(y2 - y1 < 20 or ((y2 - y1) / float(x2 - x1)) > 3.0 or ((y2 - y1) / float(x2 - x1)) < 0.3):
                #        continue
                #if(anno_cat == 'vehicle.bicycle' and self._mode == 'train'):
                #    if(y2 - y1 < 5 or ((y2 - y1) / float(x2 - x1)) > 6.0 or ((y2 - y1) / float(x2 - x1)) < 0.3):
                #        continue
                #if(anno_cat == 'human.pedestrian' and self._mode == 'train'):
                #    if(y2 - y1 < 5 or ((y2 - y1) / float(x2 - x1)) > 7.0 or ((y2 - y1) / float(x2 - x1)) < 1):
                #        continue
                cat.append(anno_cat)
                gt_classes[ix] = cls
                #overlaps is (NxM) where N = number of GT entires and M = number of classes
                overlaps[ix, cls] = 1.0
                #seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)
                ix = ix + 1
            else:
                #print(line)
                #ignore[ix] = True
                boxes_dc[ix_dc, :] = bbox
                ix_dc = ix_dc + 1
            
        if(ix == 0 and remove_without_gt is True):
            print('removing pc {} with no GT boxes specified'.format(pc_labels['assoc_frame']))
            return None
        overlaps = scipy.sparse.csr_matrix(overlaps)
        #TODO: Double return
        return {
            'pc_idx':      pc_idx,
            'scene_idx':   scene_idx,
            'scene_desc':  scene_desc,
            'filename':    filename,
            'ignore':      ignore[0:ix],
            'det':         ignore[0:ix].copy(),
            'cat':         cat,
            'difficulty':  difficulty,
            'hit':         ignore[0:ix].copy(),
            'boxes':       boxes[0:ix],
            'ids':         ids[0:ix],
            'pts':         num_pts[0:ix],
            'boxes_dc':    boxes_dc[0:ix_dc],
            'gt_classes':  gt_classes[0:ix],
            'gt_overlaps': overlaps[0:ix],
            'flipped':     False
        }

        #Post Process Step
        #filtered_boxes      = np.zeros((ix, 4), dtype=np.uint16)
        #filtered_boxes_dc   = np.zeros((ix_dc, 4), dtype=np.uint16)
        #filtered_cat        = []
        #filtered_gt_class   = np.zeros((ix), dtype=np.int32)
        #filtered_overlaps   = np.zeros((ix, self.num_classes), dtype=np.float32)
        #ix_filter = 0
        #Remove occluded examples
        #if(filter_boxes is True):
        #    for i in range(ix):
        #        remove = False
                #Any GT that overlaps with another
                #Pedestrians will require a larger overlap than cars.
                #Need overlap
                #OR
                #box behind is fully inside foreground object
                #for j in range(ix):
                    #if(i == j):
                    #    continue
                    #How many LiDAR points?
                    
                    #i is behind j
                    #z_diff = dists[i][0] - dists[j][0]
                    #n_diff = dists[i][1] - dists[j][1]
                    #if(boxes[i][0] > boxes[j][0] and boxes[i][1] > boxes[j][1] and boxes[i][2] < boxes[j][2] and boxes[i][3] < boxes[j][3]):
                    #    fully_inside = True
                    #else:
                    #    fully_inside = False
                    #overlap_comp(boxes[i],boxes[j])
                    #if(n_diff > 0.3 and fully_inside):
                    #    remove = True
                #for j in range(ix_dc):  
                    #i is behind j
                #    z_diff = dists[i][0] - dists_dc[j][0]
                #    n_diff = dists[i][1] - dists_dc[j][1]
                #    if(boxes[i][0] > boxes_dc[j][0] and boxes[i][1] > boxes_dc[j][1] and boxes[i][2] < boxes_dc[j][2] and boxes[i][3] < boxes_dc[j][3]):
                #        fully_inside = True
                #    else:
                #        fully_inside = False
                    #overlap_comp(boxes[i],boxes[j])
                #    if(n_diff > 0.3 and fully_inside):
                #        remove = True
                #if(remove is False):
                #    filtered_boxes[ix_filter] = boxes[i]
                #    filtered_gt_class[ix_filter] = gt_classes[i]
                #    filtered_cat.append(cat[i])
                #    filtered_overlaps[ix_filter] = overlaps[i]
                #    ix_filter = ix_filter + 1

            #if(ix_filter == 0 and remove_without_gt is True):
            #    print('removing element {}'.format(pc['token']))
            #    return None
        #else:
        #    ix_filter = ix
        #    filtered_boxes = boxes
        #    filtered_gt_class = gt_classes[0:ix]
        #    filtered_cat      = cat[0:ix]
        #    filtered_overlaps = overlaps

        #filtered_overlaps = scipy.sparse.csr_matrix(filtered_overlaps)
        #assert(len(boxes) != 0, "Boxes is empty for label {:s}".format(index))
        #return {
        #    'pc_idx':     pc_idx,
        #    'scene_idx':   scene_idx,
        #    'scene_desc':  scene_desc,
        #    'pcfile': filename,
        #    'ignore': ignore[0:ix_filter],
        #    'det': ignore[0:ix_filter].copy(),
        #    'cat': filtered_cat,
        #    'hit': ignore[0:ix_filter].copy(),
        #    'boxes': filtered_boxes[0:ix_filter],
        #    'boxes_dc': boxes_dc[0:ix_dc],
        #    'gt_classes': filtered_gt_class[0:ix_filter],
        #    'gt_overlaps': filtered_overlaps[0:ix_filter],
        #    'flipped': False,
        #    'seg_areas': seg_areas[0:ix_filter]
        #}

    def _do_python_eval(self, output_dir='output',mode='val'):
        #Not needed anymore, self._index has all files
        pcset = self._get_index_for_mode(mode)
        cachedir = os.path.join(self._devkit_path, 'cache')
        num_d_levels = 2
        #AP: Level 1, Level 2
        aps = np.zeros((len(self._classes)-1,num_d_levels))
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        #Loop through all classes
        for i, cls in enumerate(self._classes):
            if cls == 'dontcare' or cls == '__background__':
                continue
            if 'car' in cls:
                ovt = 0.7
            else:
                ovt = 0.5
            #waymo/results/comp_X_testing_class.txt
            detfile = self._get_results_file_template(mode,cls,output_dir)
            #Run waymo evaluation metrics on each pc
            rec, prec, ap = waymo_eval(
                detfile,
                self,
                pcset,
                cls,
                cachedir,
                mode,
                ovthresh=ovt,
                eval_type='bev',
                d_levels=num_d_levels)
            aps[i-1,:] = ap
            #Tell user of AP
            for j in range(0,num_d_levels):
                print(('Level {} AP for {} = {:.4f}'.format(j+1,cls,ap[j])))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print(('Mean AP = {:.4f} '.format(np.mean(aps[:]))))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

    def evaluate_detections(self, all_boxes, output_dir, mode):
        print('writing results to file...')
        self._write_lidar_results_file(all_boxes, mode)
        self._do_python_eval(output_dir, mode)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == 'dontcare'  or cls == '__background__':
                    continue
                filename = self._get_results_file_template(mode,cls,output_dir)
                os.remove(filename)

if __name__ == '__main__':
    # from datasets.pascal_voc import pascal_voc
    #d = pascal_voc('trainval', '2007')
    #res = d.roidb
    from IPython import embed

    embed()
