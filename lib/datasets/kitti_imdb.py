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
from PIL import Image, ImageDraw
import subprocess
import uuid
import traceback
from .kitti_eval import kitti_eval
from model.config import cfg
import shutil

class kitti_imdb(imdb):
    def __init__(self, mode='test',limiter=0):
        name = 'kitti'
        imdb.__init__(self, name)
        self._year = None
        self._devkit_path = self._get_default_path()
        self._data_path = self._devkit_path
        self._mode = mode
        self._image_ext = '.png'
        self._image_sub_dir = 'image_2'
        self._train_image_dir = os.path.join(self._data_path, 'training', 'image_2')
        self._val_image_dir   = os.path.join(self._data_path, 'evaluation', 'image_2')
        self._test_image_dir   = os.path.join(self._data_path, 'testing', 'image_2')
        self._imwidth = 1242
        self._imheight = 375
        self._imtype = 'png'
        self._mode = mode
        #Backwards compatibility
        self._train_sub_folder = 'training'
        self._val_sub_folder = 'evaluation'
        self._test_sub_folder = 'testing'
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
        self._train_image_index = sorted([d.replace('.png', '') for d in os.listdir(self._train_image_dir) if d.endswith('.png')])
        self._val_image_index = sorted([d.replace('.png', '') for d in os.listdir(self._val_image_dir) if d.endswith('.png')])
        self._test_image_index = sorted([d.replace('.png', '') for d in os.listdir(self._test_image_dir) if d.endswith('.png')])
        #Limiter

        assert os.path.exists(self._devkit_path), \
            'Kitti dataset path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)
    
    #Backwards compatibility with old release
    def mode_to_sub_folder(self,mode):
        if(mode == 'train'):
            return self._train_sub_folder
        elif(mode == 'val'):
            return self._val_sub_folder
        elif(mode == 'test'):
            return self._test_sub_folder
        else:
            return None


    def image_path_from_index(self, mode, index):
        """
    Construct an image path from the image's "index" identifier.
    """
        image_path = os.path.join(self._data_path, self.mode_to_sub_folder(mode), self._image_sub_dir,
                                  index + '.png')
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
        cache_file = os.path.join(self._devkit_path, 'cache', self._name + '_' + mode + '_gt_roidb.pkl')
        if(mode == 'train'):
            image_index = self._train_image_index
        elif(mode == 'val'):
            image_index = self._val_image_index
        elif(mode == 'test'):
            image_index = self._test_image_index
        else:
            image_index = None
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                try:
                    roidb = pickle.load(fid)
                except:
                    roidb = pickle.load(fid, encoding='bytes')
            print('{} gt roidb loaded from {}'.format(self._name, cache_file))
            return roidb

        gt_roidb = [
            self._load_kitti_annotation(index, mode) for index in image_index
        ]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def get_class(self,idx):
       return self._classes[idx]

    def rpn_roidb(self):
        if self._mode != 'test':
            #Generate the ground truth roi list (so boxes, overlaps) from the annotation list
            gt_roidb = self.gt_roidb(self._mode)
            print('got here')
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
    def _load_kitti_annotation(self, index, mode='train'):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        #print('loading kitti anno')
        filename = os.path.join(self._data_path, self.mode_to_sub_folder(mode), 'label_2', index + '.txt')
        img_filename = os.path.join(self._data_path, self.mode_to_sub_folder(mode), 'image_2', index + '.png')
        label_lines = open(filename, 'r').readlines()
        num_objs = len(label_lines)
        boxes      = np.zeros((num_objs, 4), dtype=np.uint16)
        boxes_dc   = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        ignore     = np.zeros((num_objs), dtype=np.bool)
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
            if(label_arr[0].strip() not in self._classes):
                #print('replacing {:s} with dont care'.format(label_arr[0]))
                label_arr[0] = 'dontcare'
            if('dontcare' not in label_arr[0].lower().strip()):
                #print(label_arr)
                cls = self._class_to_ind[label_arr[0].strip()]
                cat.append(label_arr[0].strip())
                boxes[ix, :] = [x1, y1, x2, y2]
                gt_classes[ix] = cls
                #overlaps is (NxM) where N = number of GT entires and M = number of classes
                overlaps[ix, cls] = 1.0
                seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)
                ix = ix + 1
            if('dontcare' in label_arr[0].lower().strip()):
                #print(line)
                boxes_dc[ix_dc, :] = [x1, y1, x2, y2]
                ix_dc = ix_dc + 1
                

        overlaps = scipy.sparse.csr_matrix(overlaps)
        #assert(len(boxes) != 0, "Boxes is empty for label {:s}".format(index))
        return {
            'img_index': index,
            'imagefile': img_filename,
            'det': ignore[0:ix].copy(),
            'ignore':ignore[0:ix],
            'hit': ignore[0:ix].copy(),
            'cat': cat,
            'boxes': boxes[0:ix],
            'boxes_dc': boxes_dc[0:ix_dc],
            'gt_classes': gt_classes[0:ix],
            'gt_overlaps': overlaps[0:ix],
            'flipped': False,
            'seg_areas': seg_areas[0:ix]
        }


    #TODO: make dependence of ROIDB in test mode, gone.
    def find_gt_for_img(self,imfile,mode):
        if(mode == 'train'):
            roidb = self.roidb
        elif(mode == 'val'):
            roidb = self.val_roidb
        for roi in roidb:
            if(roi['imagefile'] == imfile):
                return roi
        return None

    def draw_and_save(self,mode,image_token=None):
        datapath = os.path.join(cfg.DATA_DIR, self._name)
        out_file = os.path.join(cfg.DATA_DIR, self._name, self.mode_to_sub_folder(mode),'drawn')
        print('deleting files in dir {}'.format(out_file))
        shutil.rmtree(out_file)
        os.makedirs(out_file)
        if(mode == 'val'):
            roidb = self.val_roidb
        elif(mode == 'train'):
            roidb = self.roidb
        #print('about to draw in {} mode with ROIDB size of {}'.format(mode,len(roidb)))
        for i, roi in enumerate(roidb):
            if(i % 250 == 0):
                if(roi['flipped']):
                    outfile = roi['imagefile'].replace('/image_2','/drawn').replace('.{}'.format(self._imtype.lower()),'_flipped.{}'.format(self._imtype.lower()))
                else:
                    outfile = roi['imagefile'].replace('/image_2','/drawn')
                if(roi['boxes'].shape[0] != 0):
                    source_img = Image.open(roi['imagefile'])
                    if(roi['flipped'] is True):
                        source_img = source_img.transpose(Image.FLIP_LEFT_RIGHT)
                        text = "Flipped"
                    else:
                        text = "Normal"
                    draw = ImageDraw.Draw(source_img)
                    draw.text((0,0),text)
                    for roi_box,cat in zip(roi['boxes'],roi['cat']):
                        draw.text((roi_box[0],roi_box[1]),cat)
                        draw.rectangle([(roi_box[0],roi_box[1]),(roi_box[2],roi_box[3])],outline=(0,255,0))
                    for roi_box in roi['boxes_dc']:
                        draw.rectangle([(roi_box[0],roi_box[1]),(roi_box[2],roi_box[3])],outline=(255,0,0))
                    print('Saving drawn file at location {}'.format(outfile))
                    source_img.save(outfile,self._imtype)

    def delete_eval_draw_folder(self,im_folder,mode):
        datapath = os.path.join(cfg.DATA_DIR, self._name, self.mode_to_sub_folder(im_folder),'{}_drawn'.format(mode))
        print('deleting files in dir {}'.format(datapath))
        shutil.rmtree(datapath)
        os.makedirs(datapath)

    def draw_and_save_eval(self,imfile,roi_dets,roi_det_labels,dets,iter,mode):
        datapath = os.path.join(cfg.DATA_DIR, self._name)
        out_file = imfile.replace('/image_2/','/{}_drawn/iter_{}_'.format(mode,iter))
        source_img = Image.open(imfile)
        draw = ImageDraw.Draw(source_img)
        for class_dets in dets:
            #Set of detections, one for each class
            for det in class_dets:
                draw.rectangle([(det[0],det[1]),(det[2],det[3])],outline=(0,int(det[4]*255),0))
        for det,label in zip(roi_dets,roi_det_labels):
            if(label == 0):
                color = 0
            else:
                color = 255
            draw.rectangle([(det[0],det[1]),(det[2],det[3])],outline=(color,color,color))
        print('Saving file at location {}'.format(out_file))
        source_img.save(out_file,self._imtype)    

    def _get_kitti_results_file_template(self, mode, class_name):
        # data/waymo/results/<comp_id>_test_aeroplane.txt
        filename = 'det_' + mode + '_{:s}.txt'.format(class_name)
        path = os.path.join(self._devkit_path, 'results', filename)
        return path

    def _write_kitti_results_file(self, all_boxes, mode='train'):
        if(mode == 'train'):
            image_index = self._train_image_index
        elif(mode == 'val'):
            image_index = self._val_image_index
        elif(mode == 'test'):
            image_index = self._test_image_index
        else:
            image_index = []
        for cls_ind, cls in enumerate(self.classes):
            if cls == 'dontcare' or cls == '__background__':
                continue
            print('Writing {} KITTI results file'.format(cls))
            filename = self._get_kitti_results_file_template(mode,cls)
            with open(filename, 'wt') as f:
                #f.write('test')
                for im_ind, index in enumerate(image_index):
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

    def _do_python_eval(self, output_dir='output', mode='train'):
        if(mode == 'train'):
            image_index = self._train_image_index
        elif(mode == 'val'):
            image_index = self._val_image_index
        elif(mode == 'test'):
            image_index = self._test_image_index
        else:
            image_index = []
        #annopath is for labels and only labelled images
        annopath = os.path.join(self._devkit_path, self.mode_to_sub_folder(mode), 'label_2')
        print(annopath)
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
            detfile = self._get_kitti_results_file_template(mode,cls)
            #Run kitti evaluation metrics on each image
            rec, prec, ap = kitti_eval(
                detfile,
                annopath,
                image_index,
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
        cmd += 'kitti_eval(\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
            .format(self._devkit_path,  self._mode, output_dir)
        print(('Running:\n{}'.format(cmd)))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir, mode):
        print('writing results to file...')
        self._write_kitti_results_file(all_boxes, mode)
        self._do_python_eval(output_dir, mode)
        if self.config['matlab_eval']:
            self._do_matlab_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == 'dontcare'  or cls == '__background__':
                    continue
                filename = self._get_kitti_results_file_template(mode,cls)
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
