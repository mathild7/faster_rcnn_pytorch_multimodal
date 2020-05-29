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
import xml.etree.ElementTree as ET
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

class waymo_imdb(db):
    def __init__(self, mode='test',limiter=0, shuffle_en=True):
        name = 'waymo'
        db.__init__(self, name, mode)
        self.type = 'image'
        if(mode == 'test'):
            self._tod_filter_list = cfg.TEST.TOD_FILTER_LIST
        else:
            self._tod_filter_list = cfg.TRAIN.TOD_FILTER_LIST
        self._uncertainty_sort_type = cfg.UC.SORT_TYPE
        self._draw_width  = cfg.IM_SIZE[0]
        self._draw_height = cfg.IM_SIZE[1]
        self._imtype   = 'PNG'
        self._filetype = 'png'
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

        self._train_index = os.listdir(os.path.join(self._devkit_path,'train','images'))
        self._val_index   = os.listdir(os.path.join(self._devkit_path,'val','images'))
        self._val_index.sort(key=natural_keys)
        rand = SystemRandom()
        if(shuffle_en):
            print('shuffling image indices')
            rand.shuffle(self._val_index)
            rand.shuffle(self._train_index)
        if(limiter != 0):
            self._val_index   = self._val_index[:limiter]
            self._train_index = self._train_index[:limiter]
        assert os.path.exists(self._devkit_path), 'waymo dataset path does not exist: {}'.format(self._devkit_path)


    def path_from_index(self, mode, index):
        """
    Construct an image path from the image's "index" identifier.
    """
        image_path = os.path.join(self._devkit_path, mode, 'images', index)
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

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
        cache_file = os.path.join(self._devkit_path, 'cache', self._name + '_' + mode + '_image_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                try:
                    roidb = pickle.load(fid)
                except:
                    roidb = pickle.load(fid, encoding='bytes')
            print('{} gt roidb loaded from {}'.format(self._name, cache_file))
            return roidb
        labels_filename = os.path.join(self._devkit_path, mode,'labels/image_labels.json')
        gt_roidb = []
        with open(labels_filename,'r') as labels_file:
            data = labels_file.read()
            #print(data)
            #print(data)
            labels = json.loads(data)
            image_index = None
            sub_total   = 0
            if(mode == 'train'):
                image_index = self._train_index
            elif(mode == 'val'):
                image_index = self._val_index
            for img in image_index:
                #print(img)
                for img_labels in labels:
                    #print(img_labels['assoc_frame'])
                    if(img_labels['assoc_frame'] == img.replace('.{}'.format(self._imtype.lower()),'')):
                        img = os.path.join(mode,'images',img)
                        roi = self._load_waymo_annotation(img,img_labels,tod_filter_list=self._tod_filter_list)
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
    #def draw_and_save(self,mode,image_token=None):
    #    datapath = os.path.join(cfg.DATA_DIR, self._name)
    #    out_file = os.path.join(cfg.DATA_DIR, self._name ,mode,'drawn')
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
    #                outfile = roi['filename'].replace('/images','/drawn').replace('.{}'.format(self._imtype.lower()),'_flipped.{}'.format(self._imtype.lower()))
    #            else:
    #                outfile = roi['filename'].replace('/images','/drawn')
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

    def draw_and_save_eval(self,filename,roi_dets,roi_det_labels,dets,uncertainties,iter,mode,draw_folder=None):
        if(draw_folder is None):
            draw_folder = mode
        out_dir = os.path.join(get_output_dir(self,mode=mode),'{}_drawn'.format(draw_folder))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if(iter != 0):
            out_file = 'iter_{}_'.format(iter) + os.path.basename(filename).replace('.{}'.format(self._filetype.lower()),'.{}'.format(self._imtype.lower()))
        else:
            out_file = 'img-'.format(iter) + os.path.basename(filename).replace('.{}'.format(self._filetype.lower()),'.{}'.format(self._imtype.lower()))
        out_file = os.path.join(out_dir,out_file)
        source_img = Image.open(filename)
        draw = ImageDraw.Draw(source_img)
        #TODO: Magic numbers
        limiter = 15
        y_start = self._draw_height - 10*(limiter+2)
        #TODO: Swap axes of dets
        for j,class_dets in enumerate(dets):
            #Set of detections, one for each class
            #Ignore background
            if(j > 0):
                if(len(class_dets) > 0):
                    cls_uncertainties = self._normalize_uncertainties(class_dets,uncertainties[j])
                    det_idx = self._sort_dets_by_uncertainty(class_dets,cls_uncertainties,descending=True)
                    avg_det_string = 'image average: '
                    num_det = len(det_idx)
                    if(num_det < limiter):
                        limiter = num_det
                    else:
                        limiter = 15
                    for i,idx in enumerate(det_idx):
                        uc_gradient = int((limiter-i)/limiter*255.0)
                        det = class_dets[idx]
                        draw.rectangle([(det[0],det[1]),(det[2],det[3])],outline=(0,int(det[4]*255),uc_gradient),width=2)
                        det_string = '{:02} '.format(i)
                        if(i < limiter):
                            draw.text((det[0]+4,det[1]+4),det_string,fill=(0,int(det[4]*255),uc_gradient,255))
                        for key,val in cls_uncertainties.items():
                            if('cls' in key):
                                if(i == 0):
                                    avg_det_string += '{}: {:5.4f} '.format(key,np.mean(np.mean(val)))
                                det_string += '{}: {:5.4f} '.format(key,np.mean(val[idx]))
                            else:
                                if(i == 0):
                                    avg_det_string += '{}: {:6.3f} '.format(key,np.mean(np.mean(val)))
                                det_string += '{}: {:6.3f} '.format(key,np.mean(val[idx]))
                        det_string += 'confidence: {:5.4f} '.format(det[4])
                        if(i < limiter):
                            draw.text((0,y_start+i*10),det_string, fill=(0,int(det[4]*255),uc_gradient,255))
                    draw.text((0,self._draw_height-10),avg_det_string, fill=(255,255,255,255))
                elif(cfg.DEBUG.EN_TEST_MSG):
                    print('draw and save: No detections for image {}, class: {}'.format(filename,j))
        for det,label in zip(roi_dets,roi_det_labels):
            if(label == 0):
                color = 0
            else:
                color = 255
            draw.rectangle([(det[0],det[1]),(det[2],det[3])],outline=(color,color,color))
        print('Saving file at location {}'.format(out_file))
        source_img.save(out_file,self._imtype)

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

    #Only care about foreground classes
    def _load_waymo_annotation(self, img, img_labels, remove_without_gt=True,tod_filter_list=[],filter_boxes=False):
        filename = os.path.join(self._devkit_path, img)
        num_objs = len(img_labels['box'])

        boxes      = np.zeros((num_objs, 4), dtype=np.uint16)
        boxes_dc   = np.zeros((num_objs, 4), dtype=np.uint16)
        cat        = []
        track_ids  = []
        difficulties = np.zeros((num_objs), dtype=np.int32)
        pts        = np.zeros((num_objs), dtype=np.int32)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        ignore     = np.zeros((num_objs), dtype=np.bool)
        overlaps   = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        weather = img_labels['scene_type'][0]['weather']
        tod = img_labels['scene_type'][0]['tod']
        scene_desc = json.dumps(img_labels['scene_type'][0])
        #TODO: Magic number
        scene_idx  = int(int(img_labels['assoc_frame']) / cfg.MAX_IMG_PER_SCENE)
        img_idx    = int(int(img_labels['assoc_frame']) % cfg.MAX_IMG_PER_SCENE)
        #Removing night-time/day-time ROI's
        if(tod not in tod_filter_list):
            print('TOD {} not in specified filter list'.format(tod))
            return None
        seg_areas  = np.zeros((num_objs), dtype=np.float32)
        camera_extrinsic = img_labels['calibration'][0]['extrinsic_transform']
        camera_intrinsic = img_labels['calibration'][0]['intrinsic']
        # Load object bounding boxes into a data frame.
        ix = 0
        ix_dc = 0
        for i, bbox in enumerate(img_labels['box']):
            difficulty = img_labels['difficulty'][i]
            anno_cat   = img_labels['class'][i]
            track_id   = img_labels['id'][i]
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
            x1 = int(float(bbox['x1']))
            y1 = int(float(bbox['y1']))
            x2 = int(float(bbox['x2']))
            y2 = int(float(bbox['y2']))
            #if(y1 < 0):
            #    print('y1: {}'.format(y1))
            #if(x1 < 0):
            #    print('x1: {}'.format(x1))
            #Do bbox shifting in augmentation
            #if(x2 >= self._draw_width):
            #    x2 = self._draw_width - 1
            #if(y2 >= self._draw_height):
            #    y2 = self._draw_height - 1
            if(anno_cat != 'dontcare'):
                #print(label_arr)
                cls = self._class_to_ind[anno_cat]
                #Stop little clips from happening for cars
                boxes[ix, :] = [x1, y1, x2, y2]
                #if(anno_cat == 'vehicle.car'):
                #    #TODO: Magic Numbers

                #    if(float(x2 - x1) <= 0):
                #        continue
                #    if(y2 - y1 < 5 or ((y2 - y1) / float(x2 - x1)) > 3.0):
                #        continue
                #if(anno_cat == 'vehicle.bicycle' and self._mode == 'train'):
                #    if(y2 - y1 < 5 or ((y2 - y1) / float(x2 - x1)) > 6.0 or ((y2 - y1) / float(x2 - x1)) < 0.3):
                #        continue
                #if(anno_cat == 'human.pedestrian' and self._mode == 'train'):
                #    if(y2 - y1 < 5 or ((y2 - y1) / float(x2 - x1)) > 7.0 or ((y2 - y1) / float(x2 - x1)) < 1):
                #        continue
                cat.append(anno_cat)
                difficulties[ix] = difficulty
                track_ids.append(track_id)
                gt_classes[ix] = cls
                #overlaps is (NxM) where N = number of GT entires and M = number of classes
                overlaps[ix, cls] = 1.0
                seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)
                ix = ix + 1
            if(anno_cat == 'dontcare'):
                #print(line)
                #ignore[ix] = True
                boxes_dc[ix_dc, :] = [x1, y1, x2, y2]
                ix_dc = ix_dc + 1
            
        if(ix == 0 and remove_without_gt is True):
            print('removing image {} with no GT boxes specified'.format(img))
            return None
        overlaps = scipy.sparse.csr_matrix(overlaps)
        #TODO: Double return
        return {
            'img_idx':     img_idx,
            'scene_idx':   scene_idx,
            'scene_desc':  scene_desc,
            'filename':    filename,
            'ignore':      ignore[0:ix],
            'det':         ignore[0:ix].copy(),
            'cat':         cat,
            'hit':         ignore[0:ix].copy(),
            'ids':         track_ids[0:ix],
            'pts':         pts[0:ix],
            'difficulty':  difficulties[0:ix],
            'boxes':       boxes[0:ix],
            'boxes_dc':    boxes_dc[0:ix_dc],
            'gt_classes':  gt_classes[0:ix],
            'gt_overlaps': overlaps[0:ix],
            'flipped':     False,
            'seg_areas':   seg_areas[0:ix]
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
            #    print('removing element {}'.format(img['token']))
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
        #    'imgname':     img,
        #    'img_idx':     img_idx,
        #    'scene_idx':   scene_idx,
        #    'scene_desc':  scene_desc,
        #    'filename': filename,
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


    def _get_waymo_results_file_template(self, mode,class_name):
        # data/waymo/results/<comp_id>_test_aeroplane.txt
        filename = 'det_' + mode + '_{:s}.txt'.format(class_name)
        path = os.path.join(self._devkit_path, 'results', filename)
        return path

    def _write_waymo_results_file(self, all_boxes, mode):
        if(mode == 'val'):
            img_idx = self._val_index
        elif(mode == 'train'):
            img_idx = self._train_index
        elif(mode == 'test'):
            img_idx = self._test_index
        for cls_ind, cls in enumerate(self.classes):
            if cls == 'dontcare' or cls == '__background__':
                continue
            print('Writing {} waymo results file'.format(cls))
            filename = self._get_waymo_results_file_template(mode,cls)
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


    def _do_python_eval(self, output_dir='output',mode='val'):
        #Not needed anymore, self._image_index has all files
        #imagesetfile = os.path.join(self._devkit_path, self._mode_sub_folder + '.txt')
        if(mode == 'train'):
            imageset = self._train_index
        elif(mode == 'val'):
            imageset = self._val_index
        elif(mode == 'test'):
            imageset = self._test_index
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
            detfile = self._get_waymo_results_file_template(mode,cls)
            #Run waymo evaluation metrics on each image
            rec, prec, ap = waymo_eval(
                detfile,
                self,
                imageset,
                cls,
                cachedir,
                mode,
                ovthresh=ovt,
                eval_type='2d',
                d_levels=num_d_levels)
            aps[i-1,:] = ap
            #Tell user of AP
            for j in range(0,num_d_levels):
                print(('AP for {} = {:.4f}'.format(cls,ap[j])))
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
        self._write_waymo_results_file(all_boxes, mode)
        self._do_python_eval(output_dir, mode)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == 'dontcare'  or cls == '__background__':
                    continue
                filename = self._get_waymo_results_file_template(mode,cls)
                os.remove(filename)

if __name__ == '__main__':
    # from datasets.pascal_voc import pascal_voc
    #d = pascal_voc('trainval', '2007')
    #res = d.roidb
    from IPython import embed

    embed()
