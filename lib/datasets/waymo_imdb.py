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
import json
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
from random import SystemRandom
from shapely.geometry import MultiPoint, box
import traceback
from .waymo_eval import waymo_eval
from model.config import cfg


class waymo_imdb(imdb):
    def __init__(self, mode='test',limiter=0):
        name = 'waymo'
        imdb.__init__(self, name)
        self._train_scenes = []
        self._val_scenes = []
        self._test_scenes = []
        self._train_image_index = []
        self._val_image_index = []
        self._test_image_index = []
        self._devkit_path = self._get_default_path()
        self._mode = mode
        self._nusc = None
        self._scene_sel = True
        #For now one large cache file is OK, but ideally just take subset of actually needed data and cache that. No need to load nusc every time.

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

        self._train_image_index = os.listdir(os.path.join(self._devkit_path,'train','images'))
        self._val_image_index   = os.listdir(os.path.join(self._devkit_path,'train','images'))

        self._imwidth  = 1920
        self._imheight = 1280
        self._imtype   = 'JPEG'
        rand = SystemRandom()
        rand.shuffle(self._val_image_index)
        rand.shuffle(self._train_image_index)

        assert os.path.exists(self._devkit_path), 'waymo dataset path does not exist: {}'.format(self._devkit_path)

    """@property
    def nusc(self):
        if(self._nusc is None):
            cache_file = os.path.join(self._devkit_path, 'cache', self.name + '_dataset.pkl')

            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as fid:
                    try:
                        self._nusc = pickle.load(fid)
                    except:
                        self._nusc = pickle.load(fid, encoding='bytes')
                print('{} dataset loaded from {}'.format(self.name, cache_file))
            else:
                self._nusc = waymo(version='v1.0-trainval', dataroot=self._devkit_path, verbose=True);
                print('{} dataset saved to {}'.format(self.name, cache_file))
                with open(cache_file, 'wb') as fid:
                    pickle.dump(self._nusc, fid, pickle.HIGHEST_PROTOCOL)
            return self._nusc
        else:
            return self._nusc
    """
    def image_path_at(self, i, mode='train'):
        """
    Return the absolute path to image i in the image sequence.
    """
        if(mode == 'train'):
            return self.image_path_from_index(self._train_image_index[i])
        elif(mode == 'val'):
            return self.image_path_from_index(self._val_image_index[i])
        elif(mode == 'test'):
            return self.image_path_from_index(self._test_image_index[i])
        else:
            return None

    def image_path_from_index(self, index):
        """
    Construct an image path from the image's "index" identifier.
    """
        image_path = os.path.join(self._devkit_path, index['filename'])
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
        #for line in traceback.format_stack():
        #    print(line.strip())
        cache_file = os.path.join(self._devkit_path, 'cache', self.name + '_' + mode + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                try:
                    roidb = pickle.load(fid)
                except:
                    roidb = pickle.load(fid, encoding='bytes')
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb
        labels_file = os.path.join(self._devkit_path, 'labels',mode,'labels.json')
        labels = json.load(labels_file)
        image_index = None
        sub_total   = 0
        if(mode == 'train'):
            image_index = self._train_image_index
        elif(mode == 'val'):
            image_index = self._val_image_index
        gt_roidb = []
        for img in image_index:
            for img_labels in labels:
                if(img_labels['assoc_frame'] in img):
                    roi = self._load_waymo_annotation(img,img_labels)
                    if(roi is None):
                        sub_total += 1
                    else:
                        gt_roidb.append(roi)
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))
        return gt_roidb

    def draw_and_save(self,mode,image_token=None):
        datapath = os.path.join(cfg.DATA_DIR, 'waymo')
        out_file = os.path.join(cfg.DATA_DIR, 'waymo','samples','cam_front_drawn')
        if(mode == 'val'):
            roidb = self.val_roidb
        elif(mode == 'train'):
            roidb = self.roidb
        #print('about to draw in {} mode with ROIDB size of {}'.format(mode,len(roidb)))
        for i, roi in enumerate(roidb):
            if(i % 250 == 0):
                outfile = roi['imagefile'].replace('samples/CAM_FRONT/','samples/cam_front_drawn/{}'.format(mode))
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
                    #print('Saving file at location {}'.format(outfile))
                    source_img.save(outfile,'JPEG')

    def draw_and_save_eval(self,imfile,roi_dets,roi_det_labels,dets,iter,mode):
        datapath = os.path.join(cfg.DATA_DIR, 'waymo')
        out_file = imfile.replace('samples/CAM_FRONT/','samples/cam_front_{}/iter_{}_'.format(mode,iter))
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
        source_img.save(out_file,'JPEG')    


    def get_class(self,idx):
       return self._classes[idx]
    #UNUSED
    def rpn_roidb(self):
        if self._mode_sub_folder != 'testing':
            #Generate the ground truth roi list (so boxes, overlaps) from the annotation list
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb
    #UNUSED
    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print('loading {}'.format(filename))
        assert os.path.exists(filename), \
            'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = pickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _anno_to_2d_bbox(self,anno,pc_file,cam_front,lidar_top,ego_pose_cam,ego_pose_lidar,cam_intrinsic):
        # Make pixel indexes 0-based
        dists = []
        nusc_box = self.nusc.get_box(anno['token'])

        # Move them to the ego-pose frame.
        nusc_box.translate(-np.array(ego_pose_cam['translation']))
        nusc_box.rotate(Quaternion(ego_pose_cam['rotation']).inverse)

        # Move them to the calibrated sensor frame.
        nusc_box.translate(-np.array(cam_front['translation']))
        nusc_box.rotate(Quaternion(cam_front['rotation']).inverse)

        dists.append(np.linalg.norm(nusc_box.center))
        # Filter out the corners that are not in front of the calibrated sensor.
        #Corners is a 3x8 matrix, first four corners are the ones facing forward, last 4 are ons facing backward
        #(0,1) top, forward
        #(2,3) bottom, forward
        #(4,5) top, backward
        #(6,7) bottom, backward
        corners_3d = nusc_box.corners()
        #Getting first 4 values of Z
        dists.append(np.mean(corners_3d[2, :4]))
        # z is height of object for ego pose or lidar
        # y is height of object for camera frame
        #TODO: Discover why this is taking the Z axis
        in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
        corners_3d = corners_3d[:, in_front]
        #print(corners_3d)
        #above    = np.argwhere(corners_3d[2, :] > 0).flatten()
        #corners_3d = corners_3d[:, above]
        # Project 3d box to 2d.
        corner_coords = view_points(corners_3d, cam_intrinsic, True).T[:, :2].tolist()
        #print(corner_coords)
        # Keep only corners that fall within the image.

        polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
        img_canvas = box(0, 0, self._imwidth-1, self._imheight-1)

        if polygon_from_2d_box.intersects(img_canvas):
            img_intersection = polygon_from_2d_box.intersection(img_canvas)
            intersection_coords = np.array([coord for coord in img_intersection.exterior.coords])

            min_x = min(intersection_coords[:, 0])
            min_y = min(intersection_coords[:, 1])
            max_x = max(intersection_coords[:, 0])
            max_y = max(intersection_coords[:, 1])
            #print('contained pts {}'.format(contained_points))
            return [min_x, min_y, max_x, max_y], dists
        else:
            return None, dists

    #Only care about foreground classes
    def _load_waymo_annotation(self, img, img_labels, remove_without_gt=True):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        #print('loading waymo anno for img {}'.format(img['filename']))
        filename = os.path.join(self._devkit_path, img['filename'])
        lidar_data = self.nusc.get('sample_data', img['lidar_token'])
        pc_filename = os.path.join(self._devkit_path, lidar_data['filename'])
        #TEMP CODE
        #filename = 'samples/CAM_FRONT/n008-2018-08-28-16-16-48-0400__CAM_FRONT__1535488186612404.jpg'
        #if(img['filename'] != filename):
        #    return None
        #print(img)
        anno_token_list = img['anns']
        annos = []
        for anno in anno_token_list:
            annos.append(self.nusc.get('sample_annotation',anno))
        objects = []
        num_objs = len(annos)

        boxes      = np.zeros((num_objs, 4), dtype=np.uint16)
        boxes_dc   = np.zeros((num_objs, 4), dtype=np.uint16)
        dists      = np.zeros((num_objs, 2), dtype=np.float32)
        dists_dc   = np.zeros((num_objs, 2), dtype=np.uint16)
        cat        = []
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        ignore     = np.zeros((num_objs), dtype=np.bool)
        overlaps   = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas  = np.zeros((num_objs), dtype=np.float32)
        cam_front = self.nusc.get('calibrated_sensor', img['calibrated_sensor_token'])
        lidar_top = self.nusc.get('calibrated_sensor',lidar_data['calibrated_sensor_token'])
        ego_pose_cam  = self.nusc.get('ego_pose', img['ego_pose_token'])
        ego_pose_lidar = self.nusc.get('ego_pose',lidar_data['ego_pose_token'])
        camera_intrinsic = np.array(cam_front['camera_intrinsic'])

        # Load object bounding boxes into a data frame.
        ix = 0
        ix_dc = 0
        filter_boxes = True
        min_thresh_car = 40
        min_thresh_bike = 20
        populated_idx = []
        for anno in annos:
            box_coords, dist = self._anno_to_2d_bbox(anno,pc_filename,cam_front,lidar_top,ego_pose_cam,ego_pose_lidar,camera_intrinsic)
            visibility = int(anno['visibility_token']) 
            num_lidar_pts = int(anno['num_lidar_pts'])
            #if(box_coords is None or (visibility <= 1 and (dist[0] < 7).any())):
            #    continue
            if(box_coords is None or num_lidar_pts < 1 or visibility <= 1):
                continue
            x1 = box_coords[0]
            y1 = box_coords[1]
            x2 = box_coords[2]
            y2 = box_coords[3]
            #Multiple types of pedestrians, accept all.
            if('human.pedestrian.adult' in anno['category_name']):
                anno_cat = 'human.pedestrian'
            elif('human.pedestrian.child' in anno['category_name']):
                anno_cat = 'human.pedestrian'
            elif('human.pedestrian.construction_worker' in anno['category_name']):
                anno_cat = 'human.pedestrian'
            elif('human.pedestrian.police_officer' in anno['category_name']):
                anno_cat = 'human.pedestrian'
            elif('vehicle.emergency.ambulance' in anno['category_name']):
                anno_cat = 'vehicle.car'
            elif('vehicle.emergency.police' in anno['category_name']):
                anno_cat = 'vehicle.car'
            else:
                anno_cat = anno['category_name']
            if(anno_cat not in self._classes):
                #print('replacing {:s} with dont care'.format(label_arr[0]))
                anno_cat = 'dontcare'
            if(anno_cat != 'dontcare'):
                #print(label_arr)
                cls = self._class_to_ind[anno_cat]
                #Stop little clips from happening for cars
                if(filter_boxes):
                    if(((y2 - y1) / (x2 - x1)) > 5.0):
                        continue
                    if(anno_cat == 'vehicle.car'):
                        if(((x2 - x1) < min_thresh_car and ((y2 - y1) / (x2 - x1)) > 2) or ((y2 - y1) / (x2 - x1)) > 3.5):
                            continue
                    if(anno_cat == 'vehicle.bicycle'):
                        if((x2 - x1) < min_thresh_bike and (y2 - y1) / (x2 - x1) > 2.0):
                            continue
                boxes[ix, :] = [x1, y1, x2, y2]
                cat.append(anno_cat)
                dists[ix, :] = dist
                gt_classes[ix] = cls
                #overlaps is (NxM) where N = number of GT entires and M = number of classes
                overlaps[ix, cls] = 1.0
                seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)
                ix = ix + 1
            if(anno_cat == 'dontcare'):
                #print(line)
                ignore[ix] = True
                boxes_dc[ix_dc, :] = [x1, y1, x2, y2]
                dists_dc[ix_dc, :] = dist
                ix_dc = ix_dc + 1
        if(ix == 0 and remove_without_gt is True):
            print('removing element {}'.format(img['token']))
            return None
        #Post Process Step
        filtered_boxes      = np.zeros((ix, 4), dtype=np.uint16)
        filtered_boxes_dc   = np.zeros((ix_dc, 4), dtype=np.uint16)
        filtered_cat        = []
        filtered_gt_class   = np.zeros((ix), dtype=np.int32)
        filtered_overlaps   = np.zeros((ix, self.num_classes), dtype=np.float32)
        ix_new = 0
        #Remove occluded examples
        if(filter_boxes is True):
            for i in range(ix):
                remove = False
                #Any GT that overlaps with another
                #Pedestrians will require a larger overlap than cars.
                #Need overlap
                #OR
                #box behind is fully inside foreground object
                for j in range(ix):
                    if(i == j):
                        continue
                    #How many LiDAR points?
                    
                    #i is behind j
                    z_diff = dists[i][0] - dists[j][0]
                    n_diff = dists[i][1] - dists[j][1]
                    if(boxes[i][0] > boxes[j][0] and boxes[i][1] > boxes[j][1] and boxes[i][2] < boxes[j][2] and boxes[i][3] < boxes[j][3]):
                        fully_inside = True
                    else:
                        fully_inside = False
                    #overlap_comp(boxes[i],boxes[j])
                    if(n_diff > 0.3 and fully_inside):
                        remove = True
                for j in range(ix_dc):  
                    #i is behind j
                    z_diff = dists[i][0] - dists_dc[j][0]
                    n_diff = dists[i][1] - dists_dc[j][1]
                    if(boxes[i][0] > boxes_dc[j][0] and boxes[i][1] > boxes_dc[j][1] and boxes[i][2] < boxes_dc[j][2] and boxes[i][3] < boxes_dc[j][3]):
                        fully_inside = True
                    else:
                        fully_inside = False
                    #overlap_comp(boxes[i],boxes[j])
                    if(n_diff > 0.3 and fully_inside):
                        remove = True
                if(remove is False):
                    filtered_boxes[ix_new] = boxes[i]
                    filtered_gt_class[ix_new] = gt_classes[i]
                    filtered_cat.append(cat[i])
                    filtered_overlaps[ix_new] = overlaps[i]
                    ix_new = ix_new + 1

            if(ix_new == 0 and remove_without_gt is True):
                print('removing element {}'.format(img['token']))
                return None
        else:
            ix_new = ix
            filtered_boxes = boxes
            filtered_gt_class = gt_classes[0:ix]
            filtered_cat      = cat[0:ix]
            filtered_overlaps = overlaps

        filtered_overlaps = scipy.sparse.csr_matrix(filtered_overlaps)
        #assert(len(boxes) != 0, "Boxes is empty for label {:s}".format(index))
        return {
            'img_index': img['token'],
            'imagefile': filename,
            'ignore': ignore[0:ix_new],
            'det': ignore[0:ix_new].copy(),
            'cat': filtered_cat,
            'hit': ignore[0:ix_new].copy(),
            'boxes': filtered_boxes[0:ix_new],
            'boxes_dc': boxes_dc[0:ix_dc],
            'gt_classes': filtered_gt_class[0:ix_new],
            'gt_overlaps': filtered_overlaps[0:ix_new],
            'flipped': False,
            'seg_areas': seg_areas[0:ix_new]
        }

    def append_flipped_images(self,mode):
        if(mode == 'train'):
            num_images = len(self._roidb)
        elif(mode == 'val'):
            num_images = len(self._val_roidb)
        for i in range(num_images):
            boxes = self.roidb[i]['boxes'].copy()
            boxes_dc = self.roidb[i]['boxes_dc'].copy()
            img_token = self.roidb[i]['img_index']
            filepath  = self.roidb[i]['imagefile']
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
                'img_index': img_token,
                'imagefile': filepath,
                'boxes': boxes,
                'cat': cat,
                'boxes_dc': boxes_dc,
                'gt_classes': self.roidb[i]['gt_classes'],
                'gt_overlaps': self.roidb[i]['gt_overlaps'],
                'flipped': True
            }
            #Calls self.gt_roidb through a handler.
            self.roidb.append(entry)

    def _get_waymo_results_file_template(self, mode,class_name):
        # data/waymo/results/<comp_id>_test_aeroplane.txt
        filename = 'det_' + mode + '_{:s}.txt'.format(class_name)
        path = os.path.join(self._devkit_path, 'results', filename)
        return path

    def _write_waymo_results_file(self, all_boxes, mode):
        if(mode == 'val'):
            img_idx = self._val_image_index
        elif(mode == 'train'):
            img_idx = self._train_image_index
        elif(mode == 'test'):
            img_idx = self._test_image_index
        for cls_ind, cls in enumerate(self.classes):
            if cls == 'dontcare' or cls == '__background__':
                continue
            print('Writing {} waymo results file'.format(cls))
            filename = self._get_waymo_results_file_template(mode,cls)
            with open(filename, 'wt') as f:
                #f.write('test')
                for im_ind, img in enumerate(img_idx):
                    dets = all_boxes[cls_ind][im_ind]
                    #print('index: ' + index)
                    #print(dets)
                    if dets == []:
                        continue
                    # expects 1-based indices
                    for k in range(dets.shape[0]):
                        f.write(
                            '{:d} {:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(
                                im_ind, img['token'], dets[k, -1], dets[k, 0],
                                dets[k, 1], dets[k, 2],
                                dets[k, 3]))

    def _do_python_eval(self, output_dir='output',mode='val'):
        #Not needed anymore, self._image_index has all files
        #imagesetfile = os.path.join(self._devkit_path, self._mode_sub_folder + '.txt')
        if(mode == 'train'):
            imageset = self._train_image_index
        elif(mode == 'val'):
            imageset = self._val_image_index
        elif(mode == 'test'):
            imageset = self._test_image_index
        cachedir = os.path.join(self._devkit_path, 'cache')
        aps = np.zeros((len(self._classes)-1,3))
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        #Loop through all classes
        for i, cls in enumerate(self._classes):
            if cls == 'dontcare' or cls == '__background__':
                continue
            if 'Car' in cls:
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
                ovthresh=ovt)
            aps[i-1,:] = ap
            #Tell user of AP
            print(('AP for {} = {:.4f}'.format(cls,ap)))
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

    def _do_matlab_eval(self, output_dir='output'):
        print('-----------------------------------------------------')
        print('Computing results with the official MATLAB eval code.')
        print('-----------------------------------------------------')
        path = os.path.join(cfg.ROOT_DIR, 'lib', 'datasets',
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'waymo_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
            .format(self._devkit_path, self._get_comp_id(), self._mode_sub_folder, output_dir)
        print(('Running:\n{}'.format(cmd)))
        status = subprocess.call(cmd, shell=True)

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
