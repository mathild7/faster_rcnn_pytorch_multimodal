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
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw
import subprocess
import uuid
from shapely.geometry import MultiPoint, box
import traceback
from .nuscenes_eval import nuscenes_eval
from model.config import cfg
from nuscenes.utils.geometry_utils import view_points
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from pyquaternion.quaternion import Quaternion
class nuscenes_imdb(imdb):
    def __init__(self, mode='test'):
        name = 'nuscenes'
        imdb.__init__(self, name)
        self._train_scenes = []
        self._val_scenes = []
        self._test_scenes = []
        self._train_image_index = []
        self._val_image_index = []
        self._test_image_index = []
        self._devkit_path = self._get_default_path()
        self._data_path = self._devkit_path
        self._mode = mode
        self._num_train_images = 0
        #For now one large cache file is OK, but ideally just take subset of actually needed data and cache that. No need to load nusc every time.
        cache_file = os.path.join(cfg.DATA_DIR, 'cache', self.name + '_dataset.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                try:
                    self._nusc = pickle.load(fid)
                except:
                    self._nusc = pickle.load(fid, encoding='bytes')
            print('{} dataset loaded from {}'.format(self.name, cache_file))
        else:
            self._nusc = NuScenes(version='v1.0-trainval', dataroot=self._data_path, verbose=True);
            print('{} dataset saved to {}'.format(self.name, cache_file))
            with open(cache_file, 'wb') as fid:
                pickle.dump(self._nusc, fid, pickle.HIGHEST_PROTOCOL)

        self._classes = (
            'dontcare',  # always index 0
            'vehicle.car',
            'human.pedestrian',
            'vehicle.bicycle')

        self.config = {
            'cleanup': True,
            'matlab_eval': False,
            'rpn_file': None
        }
        self._class_to_ind = dict(
            list(zip(self.classes, list(range(self.num_classes)))))
        self._image_dir = os.path.join(self._data_path, 'samples', 'CAM_FRONT')
        if(mode == 'train'):
            self._val_scenes = create_splits_scenes()['val']
            self._train_scenes = create_splits_scenes()['train']
        else:
            self._test_scenes = create_splits_scenes()['test']
        #print(self._train_scenes)
        for rec in self._nusc.sample_data:
            if(rec['channel'] == 'CAM_FRONT' and rec['is_key_frame'] == True):
                rec_tmp = rec
                #Reverse lookup, getting the overall sample from the picture sample token, to get the scene information.
                rec_tmp['scene_name'] = self._nusc.get('scene',self._nusc.get('sample',rec['sample_token'])['scene_token'])['name']
                rec_tmp['anns'] = self._nusc.get('sample', rec['sample_token'])['anns']
                if(rec_tmp['scene_name'] in self._train_scenes):
                    self._train_image_index.append(rec)
                    self._num_train_images += 1
                elif(rec_tmp['scene_name'] in self._val_scenes):
                    self._val_image_index.append(rec)
        #Get global image info
        if(mode == 'train'):
            self._imwidth  = self._train_image_index[0]['width']
            self._imheight = self._train_image_index[0]['height']
            self._imtype   = self._train_image_index[0]['fileformat']
        assert os.path.exists(self._devkit_path), \
            'nuscenes dataset path does not exist: {}'.format(self._devkit_path)
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
        return os.path.join(cfg.DATA_DIR, 'nuscenes')

    def gt_roidb(self):
        """
    Return the database of ground-truth regions of interest.

    This function loads/saves from/to a cache file to speed up future calls.
    """
        #for line in traceback.format_stack():
        #    print(line.strip())
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
            self._load_nuscenes_annotation(img) for img in self._train_image_index
        ]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb
    def draw_and_save(self):
        datapath = os.path.join(cfg.DATA_DIR, 'nuscenes')
        out_file = os.path.join(cfg.DATA_DIR, 'nuscenes','samples','cam_front_drawn')
        for i, img in enumerate(self._train_image_index):
            if(i%50==0):
                img_token = img['token']
                for roi in self.roidb:
                    if(roi['img_index'] == img_token and roi['boxes'].shape[0] != 0):
                        source_img = Image.open(datapath + '/' + img['filename'])
                        draw = ImageDraw.Draw(source_img)
                        for roi_box in roi['boxes']:
                            draw.rectangle([(roi_box[0],roi_box[1]),(roi_box[2],roi_box[3])])
                        print('Saving file at location {}'.format(out_file+'/'+img['filename'].replace('samples/CAM_FRONT/','')))
                        source_img.save(out_file + '/' + img['filename'].replace('samples/CAM_FRONT/',''),'JPEG')    

    def get_class(self,idx):
       return self._classes[idx]

    def rpn_roidb(self):
        if self._mode_sub_folder != 'testing':
            #Generate the ground truth roi list (so boxes, overlaps) from the annotation list
            gt_roidb = self.gt_roidb()
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

    def _anno_to_2d_bbox(self,anno,cam_front,ego_pose,cam_intrinsic):
        # Make pixel indexes 0-based
        nusc_box = self._nusc.get_box(anno['token'])
        # Move them to the ego-pose frame.
        nusc_box.translate(-np.array(ego_pose['translation']))
        nusc_box.rotate(Quaternion(ego_pose['rotation']).inverse)

        # Move them to the calibrated sensor frame.
        nusc_box.translate(-np.array(cam_front['translation']))
        nusc_box.rotate(Quaternion(cam_front['rotation']).inverse)
        # Filter out the corners that are not in front of the calibrated sensor.
        corners_3d = nusc_box.corners()
        in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
        corners_3d = corners_3d[:, in_front]

        # Project 3d box to 2d.
        corner_coords = view_points(corners_3d, cam_intrinsic, True).T[:, :2].tolist()

        # Keep only corners that fall within the image.

        polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
        img_canvas = box(0, 0, self._imwidth, self._imheight)

        if polygon_from_2d_box.intersects(img_canvas):
            img_intersection = polygon_from_2d_box.intersection(img_canvas)
            intersection_coords = np.array([coord for coord in img_intersection.exterior.coords])

            min_x = min(intersection_coords[:, 0])
            min_y = min(intersection_coords[:, 1])
            max_x = max(intersection_coords[:, 0])
            max_y = max(intersection_coords[:, 1])

            return [min_x, min_y, max_x, max_y]
        else:
            return None

    #Only care about foreground classes
    def _load_nuscenes_annotation(self, img):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        #print('loading nuscenes anno')
        #filename = os.path.join(self._data_path, self._mode_sub_folder, 'label_2', index + '.txt')
        #print(img)
        anno_token_list = img['anns']
        annos = []
        for anno in anno_token_list:
            annos.append(self._nusc.get('sample_annotation',anno))
        objects = []
        num_objs = len(annos)

        boxes      = np.zeros((num_objs, 4), dtype=np.uint16)
        boxes_dc   = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps   = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas  = np.zeros((num_objs), dtype=np.float32)
        cam_front = self._nusc.get('calibrated_sensor', img['calibrated_sensor_token'])
        ego_pose  = self._nusc.get('ego_pose', img['ego_pose_token'])
        camera_intrinsic = np.array(cam_front['camera_intrinsic'])
        # Load object bounding boxes into a data frame.
        ix = 0
        ix_dc = 0
        populated_idx = []
        for anno in annos:
            box_coords = self._anno_to_2d_bbox(anno,cam_front,ego_pose,camera_intrinsic)
            if(box_coords is None):
                continue
            else:
                x1 = box_coords[0]
                y1 = box_coords[1]
                x2 = box_coords[2]
                y2 = box_coords[3]
            #Multiple types of pedestrians, accept all.
            if('human.pedestrian' in anno['category_name']):
                anno_cat = 'human.pedestrian'
            else:
                anno_cat = anno['category_name']
            if(anno_cat not in self._classes):
                #print('replacing {:s} with dont care'.format(label_arr[0]))
                anno_cat = 'dontcare'
            if(anno_cat != 'dontcare'):
                #print(label_arr)
                cls = self._class_to_ind[anno_cat]
                boxes[ix, :] = [x1, y1, x2, y2]
                gt_classes[ix] = cls
                #overlaps is (NxM) where N = number of GT entires and M = number of classes
                overlaps[ix, cls] = 1.0
                seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)
                ix = ix + 1
            if(anno_cat == 'dontcare'):
                #print(line)
                boxes_dc[ix_dc, :] = [x1, y1, x2, y2]
                ix_dc = ix_dc + 1
                

        overlaps = scipy.sparse.csr_matrix(overlaps)
        #assert(len(boxes) != 0, "Boxes is empty for label {:s}".format(index))
        return {
            'img_index' : img['token'],
            'boxes': boxes[0:ix],
            'boxes_dc' : boxes_dc[0:ix_dc],
            'gt_classes': gt_classes[0:ix],
            'gt_overlaps': overlaps[0:ix],
            'flipped': False,
            'seg_areas': seg_areas[0:ix]
        }

    def _get_nuscenes_results_file_template(self):
        # data/nuscenes/results/<comp_id>_test_aeroplane.txt
        filename = self._get_comp_id(
        ) + '_det_' + self._mode_sub_folder + '_{:s}.txt'
        path = os.path.join(self._devkit_path, 'results', filename)
        return path

    def _write_nuscenes_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == 'dontcare' or cls == '__background__':
                continue
            print('Writing {} nuscenes results file'.format(cls))
            filename = self._get_nuscenes_results_file_template().format(cls)
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
            #nuscenes/results/comp_X_testing_class.txt
            detfile = self._get_nuscenes_results_file_template().format(cls)
            #Run nuscenes evaluation metrics on each image
            rec, prec, ap = nuscenes_eval(
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
        cmd += 'nuscenes_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
            .format(self._devkit_path, self._get_comp_id(), self._mode_sub_folder, output_dir)
        print(('Running:\n{}'.format(cmd)))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir):
        print('writing results to file...')
        self._write_nuscenes_results_file(all_boxes)
        self._do_python_eval(output_dir)
        if self.config['matlab_eval']:
            self._do_matlab_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == 'dontcare'  or cls == '__background__':
                    continue
                filename = self._get_nuscenes_results_file_template().format(cls)
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
