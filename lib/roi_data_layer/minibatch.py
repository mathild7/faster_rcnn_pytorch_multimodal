# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
"""Compute minibatch blobs for training a Fast R-CNN network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import numpy as np
import numpy.random as npr
import cv2
import imgaug as ia
from copy import deepcopy
import imgaug.augmenters as iaa
from model.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob, prep_bev_map_for_blob, bev_map_list_to_blob
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw, ImageEnhance
import os
import sys
from pyntcloud import PyntCloud
from scipy.ndimage.filters import gaussian_filter
import spconv
import utils.bbox as bbox_utils
import shutil
import re
from datasets.waymo_lidb import waymo_lidb
from utils.kitti_utils import Calibration as kitti_calib
import utils.CADC_utils as CADC_utils
import utils.kitti_utils as kitti_utils

def draw_and_save_image_minibatch(blobs,cnt):

    datapath = os.path.join(cfg.ROOT_DIR, 'debug')
    if not os.path.exists(datapath):
        os.makedirs(datapath)
    #datapath = os.path.join(cfg.DATA_DIR, 'waymo','tmp_drawn')
    out_file = os.path.basename(blobs['filename'])
    out_file = os.path.join(datapath,out_file).replace('.png','_{}.png'.format(cnt))
    img = blobs['data'][0]*cfg.PIXEL_STDDEVS + cfg.PIXEL_MEANS
    img = img[:,:,cfg.PIXEL_ARRANGE_BGR]
    img = img.astype(dtype='uint8')
    source_img = Image.fromarray(img)
    draw = ImageDraw.Draw(source_img)
    for det in blobs['gt_boxes']:
        color = int(255*det[4])
        draw.rectangle([(det[0],det[1]),(det[2],det[3])],outline=(color,color,color))
    print('Saving file at location {}'.format(out_file))
    source_img.save(out_file,'PNG')  

def draw_and_save_lidar_minibatch(blob,cnt):
    filename = blob['filename']
    info = blob['info']
    scale = info[6]
    #lidb = waymo_lidb()
    #Extract voxel grid size
    #width   = int(info[1] - info[0])
    #Y is along height axis in image domain
    #height  = int(info[3] - info[2])
    #lidb._imheight = height
    #lidb._imwidth  = width
    datapath = os.path.join(cfg.ROOT_DIR, 'debug')
    if not os.path.exists(datapath):
        os.makedirs(datapath)
    #out_file = filename.replace('/point_clouds/','/minibatch_drawn/').replace('.{}'.format('npy'),'.{}'.format('png'))
    filenum = re.sub('[^0-9]','',(os.path.basename(filename)))
    out_file  = os.path.join(datapath,'{}_{}_minibatch.png'.format(cnt,filenum))
    #source_bin = np.load(filename)

    #draw_file  = Image.new('RGB', (width, height), (255,255,255))
    #draw = ImageDraw.Draw(draw_file)
    #lidb.draw_bev(source_bin,draw)
    #for bbox in blob['gt_boxes']:
    #    lidb.draw_bev_bbox(draw,bbox,transform=False)
    #draw_file.save(out_file.replace('.png','_bev.png'),'png')
    voxel_grid = blob['data'][0]
    voxel_grid_rgb = np.zeros((voxel_grid.shape[0],voxel_grid.shape[1],3))
    #voxel_grid_rgb.fill(255)
    voxel_grid_rgb[:,:,0] = np.max(voxel_grid[:,:,0:cfg.LIDAR.NUM_SLICES],axis=2)
    max_height = np.max(voxel_grid_rgb[:,:,0])
    min_height = np.min(voxel_grid_rgb[:,:,0])
    #min_height  = 5
    #print(max_height)
    #print(min_height)
    voxel_grid_rgb[:,:,0] = np.clip(voxel_grid_rgb[:,:,0]*(255/(max_height - min_height)),0,255)
    #voxel_grid_rgb[:,:,1] = np.clip(voxel_grid_rgb[:,:,0]*(255/(max_height - min_height)),0,255)
    #voxel_grid_rgb[:,:,2] = np.clip(voxel_grid_rgb[:,:,0]*(255/(max_height - min_height)),0,255)
    voxel_grid_rgb[:,:,1] = voxel_grid[:,:,cfg.LIDAR.NUM_SLICES]*(255/voxel_grid[:,:,cfg.LIDAR.NUM_SLICES].max())
    voxel_grid_rgb[:,:,2]  = voxel_grid[:,:,cfg.LIDAR.NUM_SLICES+1]*(255/voxel_grid[:,:,cfg.LIDAR.NUM_SLICES+1].max())
    voxel_grid_rgb        = voxel_grid_rgb.astype(dtype='uint8')
    img = Image.fromarray(voxel_grid_rgb,'RGB')
    draw = ImageDraw.Draw(img)
    if(blob['flipped'] is True):
        draw.text((0,0),'flipped')
    else:
        draw.text((0,0),'normal')
    for bbox in blob['gt_boxes']:
        #bbox[0:2] = bbox[0:2]*scale
        #bbox[3:5] = bbox[3:5]*scale
        bbox_utils.draw_bev_bbox(draw,bbox,[voxel_grid.shape[1], voxel_grid.shape[0], cfg.LIDAR.Z_RANGE[1]-cfg.LIDAR.Z_RANGE[0]],transform=False)
    #for bbox_dc in enumerate(blob['gt_boxes_dc']):
    #    lidb.draw_bev_bbox(draw,bbox_dc)
    print('Saving BEV map file at location {}'.format(out_file))
    img.save(out_file,'png')

def get_minibatch(roidb, num_classes, augment_en,cnt):
    num_frames = len(roidb)
    assert num_frames == 1, "Single batch only"
    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(
        0, high=len(cfg.TRAIN.SCALES), size=num_frames)
    assert(cfg.TRAIN.BATCH_SIZE % num_frames == 0), \
      'num_frames ({}) must divide BATCH_SIZE ({})'. \
      format(num_frames, cfg.TRAIN.BATCH_SIZE)

    # Get the input image blob, formatted for caffe
    scale = cfg.TRAIN.SCALES[random_scale_inds[0]]
    if(cfg.NET_TYPE == 'image'):
        return get_image_minibatch(roidb,num_classes,augment_en,scale,cnt)
    elif(cfg.NET_TYPE == 'lidar'):
        return get_lidar_minibatch(roidb,num_classes,augment_en,scale,cnt)
    else:
        print('getting minibatch failed. Invalid NET TYPE in cfg')
        return None

def get_lidar_minibatch(roidb, num_classes, augment_en, scale, cnt):
    """Given a roidb, construct a minibatch sampled from it."""

    gt_box_size = cfg.LIDAR.NUM_BBOX_ELEM + 1  #BBox + Cls
    #X1,Y1,Z1,X2,Y2,Z2
    area_extents = [cfg.LIDAR.X_RANGE[0],cfg.LIDAR.Y_RANGE[0],cfg.LIDAR.Z_RANGE[0],cfg.LIDAR.X_RANGE[1],cfg.LIDAR.Y_RANGE[1],cfg.LIDAR.Z_RANGE[1]]

    # Get the input lidar blob
    infos, pc_blob, local_roidb = _get_lidar_blob(roidb, area_extents, scale, augment_en)
    info = infos[0]
    roi_entry = local_roidb[0]
    #Create numpy array storage for bounding boxes (enforce type)
    gt_len  = roi_entry['boxes'].shape[0]
    dc_len  = roi_entry['boxes_dc'].shape[0]
    gt_boxes = np.empty((gt_len, gt_box_size), dtype=np.float32)
    gt_boxes_dc = np.empty((dc_len, gt_box_size), dtype=np.float32)

    #Contains point cloud tensor
    blobs = {'data': pc_blob}
    blobs['flipped'] = roi_entry['flipped']
    #assert len(pc_scales) == 1, "Single batch only"
    assert len(local_roidb) == 1, "Single batch only"

    # gt boxes: (xc, yc, zc, xd, yd, zd, theta, cls)
    gt_inds = np.where(roi_entry['ignore'] == 0)[0]
    blobs['filename'] = roi_entry['filename']
    #print(blobs['filename'])
    #TODO: Ground plane estimation and subtraction
    #Transform into voxel_grid form (flip y-axis, scale to image size (e.g. 800,700))
    gt_boxes[:, 0:-1] = bbox_utils.bbox_pc_to_voxel_grid(roi_entry['boxes'][gt_inds, :],area_extents,info)
    gt_boxes[:, 0:2] = gt_boxes[:, 0:2] * scale
    gt_boxes[:, 3:5] = gt_boxes[:, 3:5] * scale
    #shift gt_boxes to voxel domain
    bbox_labels = roi_entry['gt_classes'][gt_inds]
    gt_boxes[:, -1] = bbox_labels
    blobs['gt_boxes'] = gt_boxes
    #Do we include don't care areas, so we ignore certain ground truth boxes (Might be kitti only, even tho waymo has NLZ)
    if cfg.TRAIN.IGNORE_DC:
        gt_ind_dc = np.arange(dc_len)
        gt_boxes_dc[:, 0:-1] = roi_entry['boxes_dc'][gt_ind_dc, :]
        gt_boxes_dc[:, -1] = np.zeros(dc_len)
    #TODO: FIX
    #vg_boxes_dc = bbox_utils.bbox_pc_to_voxel_grid(gt_boxes_dc,area_extents,info)
    vg_boxes_dc = np.empty(0)
    blobs['gt_boxes_dc'] = vg_boxes_dc * scale
    blobs['info'] = np.array(np.hstack((info,scale)), dtype=np.float32)
    #blobs['info'] = np.array([pc_blob.shape[0], pc_blob.shape[1], pc_blob.shape[2]], dtype=np.float32)
    if(cfg.DEBUG.DRAW_MINIBATCH and cfg.DEBUG.EN):
        draw_and_save_lidar_minibatch(blobs,cnt)
    if(len(blobs['gt_boxes']) == 0):
        #print('No GT boxes for augmented image. Skipping')
        return None

    return blobs

def get_image_minibatch(roidb, num_classes, augment_en, scale, cnt):
    """Given a roidb, construct a minibatch sampled from it."""

    infos, im_blob, local_roidb = _get_image_blob(roidb, scale, augment_en)

    #Only one frame per minibatch allowed
    info = infos[0]
    im_scale = info[6]
    roi_entry = local_roidb[0]

    blobs = {'data': im_blob}
    blobs['info'] = info


    # gt boxes: (x1, y1, x2, y2, cls)
    #gt_inds = np.where(local_roidb[0]['gt_classes'] != 0)[0]
    #print(local_roidb[0]['ignore'])
    #TODO: Could remove.. or find some way to keep difficulty -1 seperate
    gt_inds = np.where(roi_entry['ignore'] == 0)[0]
    dc_len  = roi_entry['boxes_dc'].shape[0]
    blobs['filename'] = roi_entry['filename']
    #print('from get_image_minibatch')
    #print(blobs['filename'])
    gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
    #print('scaling gt boxes by {}'.format(im_scales[0]))
    gt_boxes[:, 0:4] = roi_entry['boxes'][gt_inds, :] * im_scale
    gt_boxes[:, 4] = roi_entry['gt_classes'][gt_inds]
    blobs['gt_boxes'] = gt_boxes
    gt_boxes_dc = np.empty((dc_len, 5), dtype=np.float32)
    if cfg.TRAIN.IGNORE_DC:
        gt_ind_dc = np.arange(dc_len)
        gt_boxes_dc[:, 0:4] = roi_entry['boxes_dc'][gt_ind_dc, :] * im_scale
        gt_boxes_dc[:, 4] = np.zeros(dc_len)
    blobs['gt_boxes_dc'] = gt_boxes_dc

    if(cfg.DEBUG.DRAW_MINIBATCH and cfg.DEBUG.EN):
        draw_and_save_image_minibatch(blobs,cnt)
    #print('gt boxes')
    #assert(len(blobs['gt_boxes']) != 0), 'gt_boxes is empty for image {:s}'.format(roi_entry['filename'])
    if(len(blobs['gt_boxes']) == 0):
        #print('No GT boxes for augmented image. Skipping')
        return None
    #assert((item is False for item in roidb[0]['ignore']).any(), 'All GT boxes are set to ignore.')
    return blobs

def filter_points(pc):
    pc_min_thresh = pc[(pc[:,0] >= cfg.LIDAR.X_RANGE[0]) & (pc[:,1] >= cfg.LIDAR.Y_RANGE[0]) & (pc[:,2] >= cfg.LIDAR.Z_RANGE[0])]
    pc_min_and_max_thresh = pc_min_thresh[(pc_min_thresh[:,0] < cfg.LIDAR.X_RANGE[1]) & (pc_min_thresh[:,1] < cfg.LIDAR.Y_RANGE[1]) & (pc_min_thresh[:,2] < cfg.LIDAR.Z_RANGE[1])]
    return pc_min_and_max_thresh

def _get_lidar_blob(roidb, pc_extents, scale, augment_en=False,mode='train'):
    """Builds an input blob from the images in the roidb at the specified
  scales.
  """
    processed_frames = []
    num_frame = len(roidb)
    infos = []
    for i in range(num_frame):
        if(mode == 'test'):
            #assert augment_en is False
            filen       = roidb[0]
            local_roidb = None
        else:
            filen = roidb[i]['filename']
        if('.bin' in filen):
            points = np.fromfile(filen, dtype=np.float32).reshape(-1, 4)
            if(cfg.DB_NAME == 'cadc'):
                calib_file = filen.replace('point_clouds','calib').replace('.bin','.txt')
                pts_rect   = CADC_utils.project_pts(calib_file,points[:, 0:3])
                #pts_rect = calib.project_velo_to_rect(points[:, 0:3])
                fov_flag = get_fov_flag(pts_rect, cfg.CADC.IMG_SIZE)
                pts_fov = points[fov_flag]
            elif(cfg.DB_NAME == 'kitti'):
                calib_file = filen.replace('velodyne','calib').replace('.bin','.txt')
                calib = kitti_calib(calib_file)
                pts_rect = calib.project_velo_to_rect(points[:, 0:3])
                fov_flag = get_fov_flag(pts_rect, cfg.KITTI.IMG_SIZE, calib)
                pts_fov = points[fov_flag]
            else:
                pts_fov = points
            #source_bin = kitti_utils.project_velo_to_rect(source_bin,calib)
            source_bin = filter_points(pts_fov)
        elif('.npy' in filen):
            source_bin = np.load(filen)
            source_bin = filter_points(source_bin)
            #print(np.max(source_bin[:,2]))
        else:
            print('Cannot handle this type of binary file')
        local_roidb = deepcopy(roidb)
        #np.random.shuffle(source_bin)
        #print('augmenting image {}'.format(roidb[i]['filename']))
        #shape 0 -> height
        #shape 1 -> width
        #TODO: Needs to be a real randomly seeded number
        #Determine params
        if(mode != 'test' and local_roidb[i]['flipped'] is True):
            print('something wrong has happened')
        np.random.seed(int.from_bytes(os.urandom(4), sys.byteorder))
        flip_frame_y  = False
        flip_frame_x  = False
        rain_sim      = False
        gauss_distort = False
        rand_dropout  = False
        if(cfg.LIDAR.SHUFFLE_PC):
            source_bin = np.random.shuffle(source_bin)

        if(augment_en):
            local_roidb[i]['flipped'] = False
            flip_frame_y  = np.random.choice([True,False],p=[0.5,0.5])
            flip_frame_x  = np.random.choice([True,False],p=[0.5,0.5])
            gauss_distort = np.random.choice([True,False],p=[0.3,0.7])
            rand_dropout  = np.random.choice([True,False],p=[0.3,0.7])

        if(flip_frame_y):
            #print('performing flip')
            #Flip source binary across Y plane
            source_bin[:,1]       = -source_bin[:,1]
            local_roidb[i]['flipped'] = True
            oldy_c = local_roidb[i]['boxes'][:, 1].copy()
            old_ry = local_roidb[i]['boxes'][:, 6].copy()
            y_mean = (cfg.LIDAR.Y_RANGE[0]+cfg.LIDAR.Y_RANGE[1])/2
            local_roidb[i]['boxes'][:, 1] = -(oldy_c-y_mean) + y_mean
            local_roidb[i]['boxes'][:, 6] = -old_ry

        if(flip_frame_x):
            #print('performing flip')
            #Flip source binary across Y plane
            source_bin[:,0] = -source_bin[:,0] + cfg.LIDAR.X_RANGE[1]
            local_roidb[i]['flipped'] = True
            oldx_c = local_roidb[i]['boxes'][:, 0].copy()
            old_ry = local_roidb[i]['boxes'][:, 6].copy()
            x_mean = (cfg.LIDAR.X_RANGE[0]+cfg.LIDAR.X_RANGE[1])/2
            local_roidb[i]['boxes'][:, 0] = -(oldx_c-x_mean) + x_mean
            local_roidb[i]['boxes'][:, 6] = -old_ry

        if(gauss_distort):
            sigma_x = np.random.uniform(0.0,0.07)
            sigma_y = np.random.uniform(0.0,0.07)
            sigma_z = np.random.uniform(0.0,0.05)
            #print('performing gauss distort sigma {} {} {}'.format(sigma_x,sigma_y,sigma_z))
            x_shift = np.random.normal(0,sigma_x,size=(source_bin.shape[0]))
            y_shift = np.random.normal(0,sigma_y,size=(source_bin.shape[0]))
            z_shift = np.random.normal(0,sigma_z,size=(source_bin.shape[0]))
            source_bin[:,0] += x_shift
            source_bin[:,1] += y_shift
            source_bin[:,2] += z_shift

        if(rand_dropout):
            pKeep = np.random.uniform(0.8,1.0)
            #print('performing random dropout {}'.format(pKeep))
            keep = pKeep > np.random.rand(source_bin.shape[0])
            source_bin = source_bin[keep]
            #source_bin = np.random.shuffle(source_bin)
            #max_ind    = source_bin.shape[0]*(1-0.9)
            #source_bin = source_bin[:max_ind]

        if(mode == 'test' and cfg.TEST.RAIN_SIM_EN):
            z = np.sqrt(np.sum(np.power(source_bin[:,0:3],2),axis=1))
            z_max = cfg[cfg.DB_NAME.upper()].LIDAR_MAX_RANGE
            #DEFINE CONSTANTS
            rho = 0.9/np.pi
            R = np.power(cfg.TEST.RAIN_RATE,0.6)
            p_min = rho/(np.pi*z_max*z_max)
            #ranges from 0% to 2% 
            sigma = 0.02*z*np.power((1-np.exp(-cfg.TEST.RAIN_RATE)),2)
            mu    = np.zeros(sigma.shape[0],)
            #p_n(z) = rho/(z^2)*exp(-0.02R^(0.6)z)
            #shift z according to:
            #z_s = z + Normal(0,sigma)
            rand_shift = np.random.normal(mu,sigma)
            z = z + rand_shift
            #Update z in original array
            source_bin[:,0:3] += np.repeat(rand_shift[:,np.newaxis],3,axis=1)/3.0
            #alpha = 0.01, beta = 0.6
            delta = np.exp(-2*0.01*R*z)
            p_n = (rho/(z*z + np.finfo(np.float64).eps))*delta
            #Attenuate intensity return value (P0*exp(-2*alpha*R^(beta)*z))
            source_bin[:,3] = source_bin[:,3]*delta
            #Remove all pts that have attenuated away
            keep_inds = np.where(p_n >= p_min)
            source_bin = source_bin[keep_inds]
        if(mode == 'test' and cfg.TEST.DROPOUT_EN):
            pKeep = 0.8
            keep = pKeep > np.random.rand(source_bin.shape[0])
            source_bin = source_bin[keep]
        #print(roidb[i]['filename'])
        #print('min z value: {}'.format(np.amin(source_bin[:,2])))
        voxel_len = cfg.LIDAR.VOXEL_LEN/scale
        num_x_voxel = int((cfg.LIDAR.X_RANGE[1] - cfg.LIDAR.X_RANGE[0])*(1/voxel_len))
        num_y_voxel = int((cfg.LIDAR.Y_RANGE[1] - cfg.LIDAR.Y_RANGE[0])*(1/voxel_len))
        num_z_voxel = int(cfg.LIDAR.NUM_SLICES)
        infos.append([0,num_x_voxel,0,num_y_voxel,0,num_z_voxel,scale])
        vertical_voxel_size = (cfg.LIDAR.Z_RANGE[1] - cfg.LIDAR.Z_RANGE[0])/(cfg.LIDAR.NUM_SLICES+0.0)

        #Shift up to have voxel grid be at bottom of pc_extents
        pc_extents[5] -= pc_extents[2]
        pc_extents[2] = 0
        assert vertical_voxel_size == cfg.LIDAR.VOXEL_HEIGHT
        voxel_generator = spconv.utils.VoxelGeneratorV2(
            voxel_size=[voxel_len, voxel_len, cfg.LIDAR.VOXEL_HEIGHT],
            point_cloud_range=pc_extents,
            max_num_points=cfg.LIDAR.MAX_PTS_PER_VOXEL,
            max_voxels=cfg.LIDAR.MAX_NUM_VOXEL
        )
        #Coords returns zyx format
        #Subtract min height, so (0m,6m) instead of (-3m,3m)
        source_bin[:,2] -= cfg.LIDAR.Z_RANGE[0]
        res = voxel_generator.generate(source_bin)
        voxels = res['voxels']
        coords = res['coordinates']
        num_points_per_voxel = res['num_points_per_voxel']
        #Generate empty numpy arra to be populated
        bev_map = np.zeros((int(num_x_voxel),int(num_y_voxel),(cfg.LIDAR.NUM_CHANNEL)),dtype=np.float32)
        #zyx to xyz
        coords[:,[2,1,0]] = coords[:,[0,1,2]]
        xy_coords = coords[:,0:2]
        #Voxel contains (x,y,z,intensity,elongation)   
        voxel_min_heights = (coords[:,2]/cfg.LIDAR.VOXEL_HEIGHT)
        voxel_min_heights = np.repeat(voxel_min_heights[:,np.newaxis], voxels.shape[1],axis=1)   
        voxel_heights = voxels[:,:,2]
        voxel_max_height = np.amax(voxel_heights, axis=1) - coords[:,2]*cfg.LIDAR.VOXEL_HEIGHT
        voxel_mh_mean    = np.mean(voxel_max_height)
        #opt = np.get_printoptions()
        #np.set_printoptions(threshold=np.inf)
        #print(voxel_max_height)
        #np.set_printoptions(**opt)
        #voxel_min_height = np.amin(voxel_heights, axis=1)
        #print('min height of frame: {}'.format(voxel_min_height))
        #Scatter height slices into bev_map
        maxheight_tuple = tuple(zip(*coords))
        bev_map[maxheight_tuple] = voxel_max_height
        #Scatter intensity into bev_map

        if(cfg.LIDAR.NUM_META_CHANNEL >= 1):
            voxel_density    = num_points_per_voxel/cfg.LIDAR.MAX_PTS_PER_VOXEL
            voxel_d_mean     = np.mean(voxel_density)
            #Scatter density into bev_map
            density_loc       = np.full((xy_coords.shape[0],1),cfg.LIDAR.NUM_SLICES)
            density_coords    = np.hstack((xy_coords,density_loc))
            density_tuple     = tuple(zip(*density_coords))
            bev_map[density_tuple] = voxel_density

        if(cfg.LIDAR.NUM_META_CHANNEL >= 2):
            voxel_intensity = np.sum(voxels[:,:,3], axis=1)/num_points_per_voxel
            intensity_loc = np.full((xy_coords.shape[0],1),cfg.LIDAR.NUM_SLICES+1)
            intensity_coords = np.hstack((xy_coords,intensity_loc))
            intensity_tuple = tuple(zip(*intensity_coords))
            tanh_intensity = np.tanh(voxel_intensity)
            tanh_i_mean    = np.mean(tanh_intensity)
            bev_map[intensity_tuple] = tanh_intensity

        if(cfg.LIDAR.NUM_META_CHANNEL >= 3):
            #Scatter elongation into bev_map
            if(cfg.DB_NAME == 'waymo'):
                voxel_elongation = np.sum(voxels[:,:,4], axis=1)/num_points_per_voxel
            else:
                voxel_elongation = np.zeros((voxels.shape[0]))
            elongation_loc = np.full((xy_coords.shape[0],1),cfg.LIDAR.NUM_SLICES+2)
            elongation_coords = np.hstack((xy_coords,elongation_loc))
            elongation_tuple = tuple(zip(*elongation_coords))
            bev_map[elongation_tuple] = np.tanh(voxel_elongation)

        #Transpose so Y(left-right)/X(front-back) is X(left-right)/Y(front-back)
        bev_map        = np.transpose(bev_map,axes=[1,0,2])
        #proc_bev_map = prep_bev_map_for_blob(bev_map, cfg.LIDAR.MEANS, cfg.LIDAR.STDDEVS, scale)
        processed_frames.append(bev_map)
    # Create a blob to hold the input images
    blob = bev_map_list_to_blob(processed_frames)

    return infos, blob, local_roidb

def _get_image_blob(roidb, im_scale, augment_en=False, mode='train'):
    """Builds an input blob from the images in the roidb at the specified
  scales.
  """
    num_images = len(roidb)
    processed_ims = []
    im_infos = []
    for i in range(num_images):
        #TODO: Probably can remove this conditional branch by just providing the filename instead of the image itself?
        if(mode == 'test'):
            #assert augment_en is False
            im          = cv2.imread(roidb[i])
            local_roidb = None
        else:
            im          = cv2.imread(roidb[i]['filename'])
            local_roidb = deepcopy(roidb)
        img_arr  = im
        mean     = 0
        sigma    = 2
        #scale


        if(augment_en and mode != 'test'):
            #print('augmenting image {}'.format(roidb[i]['filename']))
            #shape 0 -> height
            #shape 1 -> width
            flip_num = np.random.normal(1.0, 2.0)
            if(local_roidb[i]['flipped'] is True):
                print('something wrong has happened')
            if(flip_num > 1.0):
                img_arr = img_arr[:, ::-1, :]
                local_roidb[i]['flipped'] = True
                oldx1 = local_roidb[i]['boxes'][:, 0].copy()
                oldx2 = local_roidb[i]['boxes'][:, 2].copy()
                local_roidb[i]['boxes'][:, 0] = im.shape[1] - oldx2 - 1
                local_roidb[i]['boxes'][:, 2] = im.shape[1] - oldx1 - 1
            else:
                local_roidb[i]['flipped'] = False
            #iaa.Sometimes(0.5,(iaa.CropAndPad(
            #    percent=(0, 0.1),
            #    pad_mode='constant',
            #    pad_cval=(0, 255),
            #    keep_size=True
            #))),
            seq = iaa.Sequential(
                [
                    iaa.Sometimes(0.25,(iaa.Affine(
                        scale={"x": (0.9, 1.2), "y": (0.9, 1.2)},  # scale images to 80-120% of their size, individually per axis
                        translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},  # translate by -20 to +20 percent (per axis)
                        order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                        cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                        shear=(-0.05, 0.05),
                        mode='constant'  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                    ))),
                    iaa.Sometimes(0.25,iaa.Dropout((0.01, 0.05), per_channel=0.5)),
                    #iaa.SomeOf((0,1),[
                    #    iaa.Sometimes(0.25,iaa.Multiply((0.5, 1.5), per_channel=0.5)),
                    #    iaa.Invert(0.05, per_channel=True)
                    #]),
                    #iaa.OneOf([
                    #iaa.Sometimes(0.5,iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
                    #    iaa.PiecewiseAffine(scale=(0.01, 0.05))
                    #]),
                    iaa.SomeOf((0, 2),[
                        iaa.SomeOf((0,1),([
                            iaa.GaussianBlur((0.5, 2.5)),  # blur images with a sigma between 0 and 3.0
                            iaa.AverageBlur(k=(1, 3)),  # blur image using local means with kernel sizes between 2 and 7
                            iaa.MedianBlur(k=(1, 3)),  # blur image using local medians with kernel sizes between 2 and 7
                            iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5))
                        ])),
                        iaa.AdditiveGaussianNoise(
                            loc=0,
                            scale=(0.0, 0.1*255),
                            per_channel=True),
                        iaa.AddToHueAndSaturation((-5, 5),per_channel=True),  # change hue and saturation
                    ], random_order=True)
                ], random_order=False
            )
            images_aug, bboxes_aug = seq(images=[img_arr],bounding_boxes=[local_roidb[i]['boxes']])
            img_arr = images_aug[0]
            local_roidb[i]['boxes'] = bboxes_aug[0]
            orig_height = img_arr.shape[0]
            orig_width  = img_arr.shape[1]
            img_arr     = np.minimum(img_arr,255)
            img_arr     = np.maximum(img_arr,0)
            img_arr     = img_arr.astype('uint8')
            #if(down_shift < 0):
            #    img_arr = np.pad(img_arr,((0,abs(down_shift)), (0,0), (0,0)), mode='constant',constant_values=(127))[abs(down_shift):,:,:]
            #elif(down_shift > 0):
            #    img_arr = np.pad(img_arr,((abs(down_shift),0), (0,0), (0,0)), mode='constant',constant_values=(127))[:-down_shift,:,:]
            #if(right_shift < 0):
            #    img_arr = np.pad(img_arr,((0,0), (0,abs(right_shift)), (0,0)), mode='constant',constant_values=(127))[:,abs(right_shift):,:]
            #elif(right_shift > 0):
            #    img_arr = np.pad(img_arr,((0,0), (abs(right_shift),0), (0,0)), mode='constant',constant_values=(127))[:,:-right_shift,:]
            for j, roi in enumerate(local_roidb[i]['boxes']):
                #boxes[ix, :] = [x1, y1, x2, y2]
                orig = roi
                h = roi[3] - roi[1]
                w = roi[2] - roi[0]
                roi[0] = np.minimum(np.maximum(roi[0],0),orig_width-1)
                roi[2] = np.minimum(np.maximum(roi[2],0),orig_width-1)
                roi[1] = np.minimum(np.maximum(roi[1],0),orig_height-1)
                roi[3] = np.minimum(np.maximum(roi[3],0),orig_height-1)
                #TODO: magic number
                if(roi[3] - roi[1] < 2):
                    #print('removing box y0 {} y1 {}'.format(roi[1],roi[3]))
                    local_roidb[i]['ignore'][j] = True
                #TODO: magic number
                if(roi[2] - roi[0] < 2):
                    #print('removing box  x0 {} x1 {}'.format(roi[0],roi[2]))
                    local_roidb[i]['ignore'][j] = True

                wc = roi[2] - roi[0]
                hc = roi[3] - roi[1]
                if(h != 0 and hc/h < 0.1):
                    local_roidb[i]['ignore'][j] = True
                elif(w != 0 and wc/w < 0.1):
                    local_roidb[i]['ignore'][j] = True

                if(local_roidb[i]['ignore'][j] is False and roi[2] < roi[0]):
                    print('x2 is smaller than x1')
                if(local_roidb[i]['ignore'][j] is False and roi[3] < roi[1]):
                    print('y2 is smaller than y1')
                if(local_roidb[i]['ignore'][j] is False and wc > w):
                    print('new x is larger than old x diff')
                if(local_roidb[i]['ignore'][j] is False and hc > h):
                    print('new y diff is larger than old y diff')
            im = img_arr
        elif(augment_en and mode == 'test'):
            seq = iaa.Sequential(
                [
                    iaa.weather.Rain(speed=(0.15,0.25))
                    #iaa.Dropout((0.2))
                    #iaa.GaussianBlur(sigma=(3.0,3.5))
                ], random_order=False)
            images_aug = seq(images=[img_arr])
            img_arr = images_aug[0]
            orig_height = img_arr.shape[0]
            orig_width  = img_arr.shape[1]
            img_arr     = np.minimum(img_arr,255)
            img_arr     = np.maximum(img_arr,0)
            img_arr     = img_arr.astype('uint8')
            im = img_arr
        #minibatch(im,local_roidb[0])
        #draw_and_save_minibatch(im[:,:,cfg.PIXEL_ARRANGE_BGR],roidb[i])
        #TODO: Move scaling to be before imgaug, to save time
        im = prep_im_for_blob(im, cfg.PIXEL_MEANS, cfg.PIXEL_STDDEVS, cfg.PIXEL_ARRANGE, im_scale)
            #x_min, x_max, y_min, y_max, scale
        info = np.array([0, im.shape[1], 0, im.shape[0], 0, 0, im_scale], dtype=np.float32)
        im_infos.append(info)
        processed_ims.append(im)
    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return im_infos, blob, local_roidb

def get_fov_flag(pts_rect, img_shape, calib=None):
    '''
    Valid point should be in the image (and in the PC_AREA_SCOPE)
    :param pts_rect:
    :param img_shape:
    :return:
    '''
    if(calib is not None):
        pts_img = calib.project_rect_to_image(pts_rect)
    else:
        pts_img = pts_rect
    val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
    val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
    val_flag_merge = np.logical_and(val_flag_1, val_flag_2)

    return val_flag_merge
