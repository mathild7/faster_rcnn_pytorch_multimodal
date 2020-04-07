import tensorflow as tf
import os
import math
import numpy as np
import itertools
import json
from PIL import Image, ImageDraw
from enum import Enum
tf.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

class cam_enum(Enum):
    UNKNOWN     = 0
    FRONT       = 1
    FRONT_LEFT  = 2
    FRONT_RIGHT = 3
    SIDE_LEFT   = 4
    SIDE_RIGHT  = 5

save_imgs = False
mypath = '/home/mat/thesis/data2/waymo/train'
tfrec_path = os.path.join(mypath,'compressed_tfrecords')
top_crop = 550
bbox_top_min = 30
file_list = [os.path.join(tfrec_path,f) for f in os.listdir(tfrec_path) if os.path.isfile(os.path.join(tfrec_path,f))]
#filename = 'segment-11799592541704458019_9828_750_9848_750_with_camera_labels.tfrecord'
with open(os.path.join(mypath,'labels','image_labels_new.json'), 'w') as json_file:
    json_struct = []
    for i,filename in enumerate(file_list):
        if(i > 350):
            break
        if('.tfrecord' in filename):
            print('opening {}'.format(filename))
            dataset = tf.data.TFRecordDataset(filename,compression_type='')
            for j,data in enumerate(dataset):
                if(j%3 == 0):
                    json_calib = {}
                    frame = open_dataset.Frame()
                    frame.ParseFromString(bytearray(data.numpy()))
                    #print(frame.context)
                    for calib in frame.context.camera_calibrations:
                        if(cam_enum(calib.name)  == cam_enum.FRONT):
                            json_calib['intrinsic']           = []
                            json_calib['extrinsic_transform'] = []
                            for val in calib.intrinsic:
                                json_calib['intrinsic'].append(val)
                            for val in calib.extrinsic.transform:
                                json_calib['extrinsic_transform'].append(val)
                            #print('found calibration')
                            #print(calib)
                    #TODO: Output calibration to JSON array file.
                    k_l = 0
                    for img, labels in zip(frame.images, frame.camera_labels):
                        
                        if(cam_enum(img.name)  == cam_enum.FRONT and save_imgs):
                        #print(img.DESCRIPTOR.fields)
                            im_data = tf.image.decode_jpeg(img.image, channels=3).numpy()
                            im_data = im_data[:][top_crop:][:]
                            im_data = Image.fromarray(im_data)
                            img_filename = '{0:05d}.png'.format(i*1000+j)
                            out_file = os.path.join(mypath, 'images_new',img_filename)
                            draw = ImageDraw.Draw(im_data)
                            im_data.save(out_file,'PNG')

                        if(cam_enum(labels.name) == cam_enum.FRONT):
                            json_labels = {}
                            json_labels['box']   = []
                            json_labels['class'] = []
                            json_labels['difficulty']  = []
                            json_labels['id'] = []
                            json_labels['assoc_frame'] = '{0:05d}'.format(i*1000+j) 
                            json_labels['scene_name']  = frame.context.name
                            json_labels['scene_type']  = []
                            json_labels['scene_type'].append({'weather': frame.context.stats.weather,
                                                            'tod': frame.context.stats.time_of_day})
                            #print(json_calib)
                            json_labels['calibration'] = []
                            json_labels['calibration'].append(json_calib)
                            for label in labels.labels:
                                x1 = float(label.box.center_x) - float(label.box.length)/2
                                y1 = float(label.box.center_y) - float(label.box.width)/2
                                x2 = float(label.box.center_x) + float(label.box.length)/2
                                y2 = float(label.box.center_y) + float(label.box.width)/2
                                y1 = np.maximum(y1-top_crop,0)
                                y2 = np.maximum(y2-top_crop,0)
                                if(y1 < 0):
                                    print('y1: '.format(y1))
                                if(y2 < 0):
                                    print('y2: '.format(y2))
                                if(y2-y1 <= bbox_top_min and y1 == 0):
                                    continue
                                json_labels['box'].append({
                                    'x1': '{:.2f}'.format(x1),
                                    'y1': '{:.2f}'.format(y1),
                                    'x2': '{:.2f}'.format(x2),
                                    'y2': '{:.2f}'.format(y2)
                                })
                                json_labels['id'].append(label.id)
                                json_labels['class'].append(label.type)
                                json_labels['difficulty'].append(label.detection_difficulty_level)
                            #print(json_labels)
                            #k_l = k_l + 1
                            json_struct.append(json_labels)
                #print(k)
                #k_i = 0

                #for img in frame.images:
                    #print(img)
                #    if(cam_enum(img.name)  == cam_enum.FRONT):
                       #print(img.DESCRIPTOR.fields)
                #        im_data = tf.image.decode_jpeg(img.image, channels=3).numpy()
                #        im_data = im_data[:][top_crop:][:]
                #        im_data = Image.fromarray(im_data)
                #        img_filename = '{0:05d}.png'.format(i*1000+j)
                #        out_file = os.path.join(mypath, 'images',img_filename)
                #        draw = ImageDraw.Draw(im_data)
                #        im_data.save(out_file,'PNG')
                #        k_i = k_i + 1
                #if(k_i != k_l):
                #    print('image to label mismatch!!!')
    json.dump(json_struct,json_file)
