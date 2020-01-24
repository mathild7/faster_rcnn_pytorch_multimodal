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
from utils.blob import prep_im_for_blob, im_list_to_blob
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw, ImageEnhance
import os
from scipy.ndimage.filters import gaussian_filter

def draw_and_save_minibatch(im,roidb):
    datapath = os.path.join(cfg.DATA_DIR, 'waymo','tmp_drawn')
    out_file = roidb['imgname'].replace('images/','img-')
    out_file = os.path.join(datapath,out_file)
    source_img = Image.fromarray(im)
    draw = ImageDraw.Draw(source_img)
    for det,label in zip(roidb['boxes'],roidb['ignore']):
        if(label == 0):
            color = 0
        else:
            color = 255
        draw.rectangle([(det[0],det[1]),(det[2],det[3])],outline=(color,color,color))
    print('Saving file at location {}'.format(out_file))
    source_img.save(out_file,'PNG')    

def get_minibatch(roidb, num_classes, augment_en):
    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(
        0, high=len(cfg.TRAIN.SCALES), size=num_images)
    assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
      'num_images ({}) must divide BATCH_SIZE ({})'. \
      format(num_images, cfg.TRAIN.BATCH_SIZE)

    # Get the input image blob, formatted for caffe
    im_blob, im_scales, local_roidb = _get_image_blob(roidb, random_scale_inds, augment_en)
    #print('got image {}'.format(roidb[0]['imagefile']))
    #print('token {}'.format(roidb[0]['imgname']))
    #print('is it flipped?: {}'.format(roidb[0]['flipped']))
    #Contains actual image
    blobs = {'data': im_blob}

    assert len(im_scales) == 1, "Single batch only"
    assert len(roidb) == 1, "Single batch only"

    # gt boxes: (x1, y1, x2, y2, cls)
    #gt_inds = np.where(local_roidb[0]['gt_classes'] != 0)[0]
    #print(local_roidb[0]['ignore'])
    gt_inds = np.where(local_roidb[0]['ignore'] == 0)[0]
    dc_len  = local_roidb[0]['boxes_dc'].shape[0]
    blobs['imagefile'] = local_roidb[0]['imagefile']
    gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
    #print('scaling gt boxes by {}'.format(im_scales[0]))
    gt_boxes[:, 0:4] = local_roidb[0]['boxes'][gt_inds, :] * im_scales[0]
    gt_boxes[:, 4] = local_roidb[0]['gt_classes'][gt_inds]
    blobs['gt_boxes'] = gt_boxes
    gt_boxes_dc = np.empty((dc_len, 5), dtype=np.float32)
    if cfg.TRAIN.IGNORE_DC:
        gt_ind_dc = np.arange(dc_len)
        gt_boxes_dc[:, 0:4] = local_roidb[0]['boxes_dc'][gt_ind_dc, :] * im_scales[0]
        gt_boxes_dc[:, 4] = np.zeros(dc_len)
    blobs['gt_boxes_dc'] = gt_boxes_dc
    blobs['im_info'] = np.array(
        [im_blob.shape[1], im_blob.shape[2], im_scales[0]], dtype=np.float32)
    #print('gt boxes')
    #assert(len(blobs['gt_boxes']) != 0), 'gt_boxes is empty for image {:s}'.format(local_roidb[0]['imagefile'])
    if(len(blobs['gt_boxes']) == 0):
        #print('No GT boxes for augmented image. Skipping')
        return None
    #assert((item is False for item in roidb[0]['ignore']).any(), 'All GT boxes are set to ignore.')
    return blobs


def _get_image_blob(roidb, scale_inds, augment_en=False):
    """Builds an input blob from the images in the roidb at the specified
  scales.
  """
    num_images = len(roidb)
    processed_ims = []
    im_scales = []
    for i in range(num_images):
        im = cv2.imread(roidb[i]['imagefile'])
        #im  = cv2.imread('/home/mat/black.png')
        #print(roidb[i]['imagefile'])
        #if('000318' in roidb[i]['imagefile']):
        #    print('--------------------------')
        #    print('minibatch images')
        #    print('--------------------------')

        img_arr  = im
        mean     = 0
        sigma    = 2
        local_roidb = deepcopy(roidb)
        if(augment_en):
            #print('augmenting image {}'.format(roidb[i]['imgname']))
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
                    iaa.Sometimes(0.6,(iaa.Affine(
                        scale={"x": (1, 2), "y": (1, 2)},  # scale images to 80-120% of their size, individually per axis
                        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # translate by -20 to +20 percent (per axis)
                        order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                        cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                        shear=(-0.5, 0.5),
                        mode='constant'  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                    ))),
                    #iaa.Sometimes(0.5,iaa.Dropout((0.01, 0.1), per_channel=0.5)),
                    #iaa.SomeOf((0,1),[
                    #iaa.Sometimes(0.5,iaa.Multiply((0.5, 1.5), per_channel=0.5)),
                    #    iaa.Invert(0.05, per_channel=True)
                    #]),
                    #iaa.OneOf([
                    iaa.Sometimes(0.5,iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
                    #    iaa.PiecewiseAffine(scale=(0.01, 0.05))
                    #]),
                    iaa.SomeOf((0, 2),[
                        iaa.SomeOf((0,3),([
                            iaa.GaussianBlur((0.5, 3.0)),  # blur images with a sigma between 0 and 3.0
                            iaa.AverageBlur(k=(3, 7)),  # blur image using local means with kernel sizes between 2 and 7
                            iaa.MedianBlur(k=(3, 7)),  # blur image using local medians with kernel sizes between 2 and 7
                            iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
                            iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
                        ])),
                        iaa.AdditiveGaussianNoise(
                            loc=0,
                            scale=(0.0, 0.08*255),
                            per_channel=0.5),
                        iaa.AddToHueAndSaturation((-30, 30)),  # change hue and saturation
                    ], random_order=True)
                ], random_order=False
            )
            images_aug, bboxes_aug = seq(images=[img_arr],bounding_boxes=[local_roidb[i]['boxes']])
            img_arr = images_aug[0]
            local_roidb[i]['boxes'] = bboxes_aug[0]
            #gaussian   = np.random.normal(mean, sigma, (img_arr.shape[0],img_arr.shape[1],3))
            #blur_amt   = np.random.normal(0.7, 0.1)
            #blur_amt   = np.maximum(blur_amt,0.4)
            #blur_amt   = np.minimum(blur_amt,1.0)
            orig_height = img_arr.shape[0]
            orig_width  = img_arr.shape[1]
            #brightness = np.random.normal(mean,sigma*4)
            #Up and left shift by 20px
            #right_shift = int(np.random.normal(mean,sigma*3))
            #down_shift  = int(np.random.normal(mean,sigma*3))
            #right_shift = 0
            #down_shift  = 0
            #img_arr     = img_arr.astype('float')
            #img_arr    += gaussian
            #img_arr    += brightness
            img_arr     = np.minimum(img_arr,255)
            img_arr     = np.maximum(img_arr,0)
            img_arr     = img_arr.astype('uint8')
            #img = Image.fromarray(img_arr)
            #contrast_enhancer = ImageEnhance.Contrast(img)
            #img               = contrast_enhancer.enhance(1.2)
            #blur_enhancer     = ImageEnhance.Sharpness(img)
            #img               = (blur_enhancer.enhance(blur_amt))
            #img               = img.crop((c_left,c_top,c_right,c_bottom))
            #img               = img.resize((1280,730))
            #img_arr            = np.array(img)
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
                roi[0] = np.minimum(np.maximum(roi[0],0),orig_width-1)
                roi[2] = np.minimum(np.maximum(roi[2],0),orig_width-1)
                roi[1] = np.minimum(np.maximum(roi[1],0),orig_height-1)
                roi[3] = np.minimum(np.maximum(roi[3],0),orig_height-1)
                #TODO: magic number
                if(roi[3] - roi[1] < 12 and (roi[3] >= img_arr.shape[0]-1 or roi[1] <= 0)):
                    #print('removing box y0 {} y1 {}'.format(roi[1],roi[3]))
                    local_roidb[i]['ignore'][j] = True
                #TODO: magic number
                if(roi[2] - roi[0] < 12 and (roi[2] >= img_arr.shape[1]-1 or roi[0] <= 0)):
                    #print('removing box  x0 {} x1 {}'.format(roi[0],roi[2]))
                    local_roidb[i]['ignore'][j] = True

                w = roi[2] - roi[0]
                h = roi[3] - roi[1]
                if(h < 0.1):
                    local_roidb[i]['ignore'][j] = True
                elif(w < 0.1):
                    local_roidb[i]['ignore'][j] = True
                elif(h/w > 3.5 or w/h > 5.0):
                    local_roidb[i]['ignore'][j] = True

                if(local_roidb[i]['ignore'][j] is False and roi[2] < roi[0]):
                    print('x2 is smaller than x1')
                if(local_roidb[i]['ignore'][j] is False and roi[3] < roi[1]):
                    print('y2 is smaller than y1')
                if(local_roidb[i]['ignore'][j] is False and roi[2] - roi[0] > orig[2] - orig[0]):
                    print('new x is larger than old x diff')
                if(local_roidb[i]['ignore'][j] is False and roi[3] - roi[1] > orig[3] - orig[1]):
                    print('new y diff is larger than old y diff')
            im = img_arr
        #draw_and_save_minibatch(im,local_roidb[0])
        #draw_and_save_minibatch(im[:,:,cfg.PIXEL_ARRANGE_BGR],roidb[i])
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, cfg.PIXEL_STDDEVS, cfg.PIXEL_ARRANGE, target_size,
                                        cfg.TRAIN.MAX_SIZE)
        im_scales.append(im_scale)
        processed_ims.append(im)
    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, im_scales, local_roidb
