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

class laser_enum(Enum):
    UNKNOWN     = 0
    TOP         = 1
    FRONT       = 2
    SIDE_LEFT   = 3
    SIDE_RIGHT  = 4
    REAR        = 5

def compute_2d_bounding_box(img,points):
    """Compute the 2D bounding box for a set of 2D points.
    
    img_or_shape: Either an image or the shape of an image.
                  img_or_shape is used to clamp the bounding box coordinates.
    
    points: The set of 2D points to use
    """
    shape = img.shape

    # Compute the 2D bounding box and draw a rectangle
    x1 = np.amin(points[...,0])
    x2 = np.amax(points[...,0])
    y1 = np.amin(points[...,1])
    y2 = np.amax(points[...,1])

    x1 = min(max(0,x1),shape[1])
    x2 = min(max(0,x2),shape[1])
    y1 = min(max(0,y1),shape[0])
    y2 = min(max(0,y2),shape[0])

    return (x1,y1,x2,y2)

def label_3D_to_image(camera_calib, lidar_calib, metadata, bbox):
    bbox_transform_matrix = get_box_transformation_matrix(bbox)  
    instrinsic = camera_calib['intrinsic']
    extrinsic = np.array(camera_calib['extrinsic_transform']).reshape(4,4)
    vehicle_to_image = get_image_transform(instrinsic, extrinsic)  # magic array 4,4 to multiply and get image domain
    box_to_image = np.matmul(vehicle_to_image, bbox_transform_matrix)


    # Loop through the 8 corners constituting the 3D box
    # and project them onto the image
    vertices = np.empty([2,2,2,2])
    # 1: 000, 2: 001, 3: 010:, 4: 100
    for k in [0, 1]:
        for l in [0, 1]:
            for m in [0, 1]:
                # 3D point in the box space
                v = np.array([(k-0.5), (l-0.5), (m-0.5), 1.])

                # Project the point onto the image
                v = np.matmul(box_to_image, v)

                # If any of the corner is behind the camera, ignore this object.
                if v[2] < 0:
                    return None

                vertices[k,l,m,:] = [v[0]/v[2], v[1]/v[2]]

    vertices = vertices.astype(np.int32)

    return vertices

def get_box_transformation_matrix(box):
    """Create a transformation matrix for a given label box pose."""

    tx,ty,tz = box.center_x,box.center_y,box.center_z
    c = math.cos(box.heading)
    s = math.sin(box.heading)

    sl, sh, sw = box.length, box.height, box.width

    return np.array([
        [ sl*c,-sw*s,  0,tx],
        [ sl*s, sw*c,  0,ty],
        [    0,    0, sh,tz],
        [    0,    0,  0, 1]])

def get_image_transform(intrinsic, extrinsic):
    """ For a given camera calibration, compute the transformation matrix
        from the vehicle reference frame to the image space.
    """
    # Camera model:
    # | fx  0 cx 0 |
    # |  0 fy cy 0 |
    # |  0  0  1 0 |
    camera_model = np.array([
        [intrinsic[0], 0, intrinsic[2], 0],
        [0, intrinsic[1], intrinsic[3], 0],
        [0, 0,                       1, 0]])

    # Swap the axes around
    axes_transformation = np.array([
        [0,-1,0,0],
        [0,0,-1,0],
        [1,0,0,0],
        [0,0,0,1]])

    # Compute the projection matrix from the vehicle space to image space.
    vehicle_to_image = np.matmul(camera_model, np.matmul(axes_transformation, np.linalg.inv(extrinsic)))
    return vehicle_to_image


save_imgs = True
mypath = '/home/mat/thesis/data2/waymo/train'
savepath = os.path.join(mypath,'images_3d_new')
if not os.path.isdir(savepath):
    print('making path: {}'.format(savepath))
    os.makedirs(savepath)
tfrec_path = os.path.join(mypath,'compressed_tfrecords')
top_crop = 550
bbox_top_min = 30
file_list = [os.path.join(tfrec_path,f) for f in os.listdir(tfrec_path) if os.path.isfile(os.path.join(tfrec_path,f))]
file_list = sorted(file_list)
#filename = 'segment-11799592541704458019_9828_750_9848_750_with_camera_labels.tfrecord'
with open(os.path.join(mypath,'labels','image_labels_3d_new.json'), 'w') as json_file:
    json_struct = []
    for i,filename in enumerate(file_list):        
        #if(i > 500):
        #    break
        if('.tfrecord' in filename):
            print('opening {}'.format(filename))
            dataset = tf.data.TFRecordDataset(filename,compression_type='')
            for j,data in enumerate(dataset):
                frame_idx = i*1000+j  
                if(j%10 == 0):
                    img_calib = {}
                    laser_calib = {}
                    frame = open_dataset.Frame()
                    frame.ParseFromString(bytearray(data.numpy()))
                    #print(frame.context)
                    for calib in frame.context.camera_calibrations:
                        if(cam_enum(calib.name)  == cam_enum.FRONT):
                            img_calib['intrinsic']           = []
                            img_calib['extrinsic_transform'] = []
                            for val in calib.intrinsic:
                                img_calib['intrinsic'].append(val)
                            for val in calib.extrinsic.transform:
                                img_calib['extrinsic_transform'].append(val)

                    for calib in frame.context.laser_calibrations:
                        if(laser_enum(calib.name)  == laser_enum.TOP):
                            laser_calib['beam_inclinations']   = []
                            laser_calib['beam_inclination_max'] = calib.beam_inclination_max
                            laser_calib['beam_inclination_min'] = calib.beam_inclination_min
                            laser_calib['extrinsic_transform'] = []
                            for val in calib.beam_inclinations:
                                laser_calib['beam_inclinations'].append(val)
                            ext = calib.extrinsic
                            for val in ext.transform:
                                laser_calib['extrinsic_transform'].append(val)
                            #print('found calibration')
                            #print(calib)
                    #TODO: Output calibration to JSON array file.                    
                    for img in frame.images:  #img from array and save
                        if(cam_enum(img.name)  == cam_enum.FRONT):
                            im_data = tf.image.decode_jpeg(img.image, channels=3).numpy()  #content and 3 for rgb image      
                            im_arr = im_data
                            im_data = Image.fromarray(im_data)  #pillow method
                            img_filename = '{0:07d}.png'.format(frame_idx)  # 7 numbers with frame idx
                            out_file = os.path.join(savepath, img_filename)
                            draw = ImageDraw.Draw(im_data)
                    assert im_data is not None
                    for label in frame.laser_labels:                        
                        #Transform to image domain
                        bbox = label.box
                        # bbox2D has format [x1,y1,x2,y2]
                        bbox2D = label_3D_to_image(img_calib, laser_calib, label.metadata, bbox)  
                        if(bbox2D is None):
                            continue
                        bbox2D = compute_2d_bounding_box(im_arr, bbox2D)
                        draw.rectangle(bbox2D)
                    im_data.save(out_file,'PNG')   
