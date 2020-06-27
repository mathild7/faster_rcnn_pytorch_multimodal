import _init_paths
import tensorflow as tf
import os
import math
import numpy as np
import itertools
import json
from PIL import Image, ImageDraw
from enum import Enum
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import transform_utils
from multiprocessing import Process, Pool
import multiprocessing.managers
from concurrent.futures import ProcessPoolExecutor, as_completed
from model.config import cfg
import utils.bbox as bbox_utils
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

tf.enable_eager_execution()

top_crop = 0
bot_crop = 0
bbox_edge_thresh = 10
lidar_thresh_dist = 30
save_imgs = True
mypath = '/home/mat/thesis/data2/waymo/val'
img_savepath = os.path.join(mypath,'images_new')
if not os.path.isdir(img_savepath):
    print('making path: {}'.format(img_savepath))
    os.makedirs(img_savepath)
lidar_savepath = os.path.join(mypath,'point_clouds_new')
if not os.path.isdir(lidar_savepath):
    print('making path: {}'.format(lidar_savepath))
    os.makedirs(lidar_savepath)
skip_binaries = False
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

def main():
    tfrecord_path = mypath + '/compressed_tfrecords'
    #savepath = os.path.join(mypath,'point_clouds_new')
    #if not os.path.isdir(savepath):
    #    print('making path: {}'.format(savepath))
    #    os.makedirs(savepath)
    num_proc = 16
    #top_crop = 550
    bbox_top_min = 30
    file_list = [os.path.join(tfrecord_path,f) for f in os.listdir(tfrecord_path) if os.path.isfile(os.path.join(tfrecord_path,f))]
    file_list = sorted(file_list)
    #filename = 'segment-11799592541704458019_9828_750_9848_750_with_camera_labels.tfrecord'
    #pool = Pool(processes=16)
    #m = 
    with open(os.path.join(mypath,'labels','combined_labels_new.json'), 'w') as json_file:
        json_struct = []
        for i,filename in enumerate(file_list):
            #if(i > 25):
            #    break
            if('tfrecord' in filename):
                print('opening {}'.format(filename))
                dataset = tf.data.TFRecordDataset(filename,compression_type='')
                dataset_list = []
                for elem in dataset:
                    dataset_list.append(elem)
                dataset_len = len(dataset_list)
                for j in range(0,dataset_len):
                    if(j%10 == 0):
                        frame = open_dataset.Frame()
                        frame.ParseFromString(bytearray(dataset_list[j].numpy()))
                        proc_data = (i,j,frame,mypath)
                        #print('frame: {}'.format(i*1000+j))
                        labels = frame_loop(proc_data)
                        #print(len(labels['box']))
                        json_struct.append(labels)
        json.dump(json_struct,json_file)


def frame_loop(proc_data):
    import tensorflow as tfp
    tfp.enable_eager_execution()
    (i,j,frame,mypath) = proc_data
    #if(len(frame.no_label_zones) != 0):
    #    print('found a NLZ')
    frame_idx = i*1000+j
    if(not skip_binaries):

        (range_images, range_image_top_pose) = parse_range_image(frame,tfp)
        points                = convert_range_image_to_point_cloud(frame,range_images,range_image_top_pose, tfp)
        points_top            = points[laser_enum.TOP.value-1]
        points_top_filtered   = filter_points(points_top)
        points_2              = convert_range_image_to_point_cloud(frame,range_images,range_image_top_pose, tfp, ri_index=1)
        points_top_2          = points_2[laser_enum.TOP.value-1]
        # points_top_filtered_2 = points_top_2[laser_enum.TOP.value-1]
        points_top_filtered_2   = filter_points(points_top_2)
        bin_filename = '{0:07d}.npy'.format(frame_idx)
        out_file     = os.path.join(lidar_savepath,bin_filename)
        if(len(points_top_filtered) > 0):
            np.save(out_file,points_top_filtered)
            #fp = open(out_file, 'wb')
            #fp.write(bytearray(points_top_filtered))
            #fp.close()
        else:
            print('Lidar pointcloud {} is empty'.format(bin_filename))
            #    k_i = k_i + 1
        #if(k_i != k_l):
        #    print('image to label mismatch!!!')
        for img in frame.images:
            if(cam_enum(img.name)  == cam_enum.FRONT):
                #print(img.DESCRIPTOR.fields)
                im_arr = cv2.imdecode(np.frombuffer(img.image, np.uint8), cv2.IMREAD_COLOR)
                im_arr = cv2.cvtColor(im_arr, cv2.COLOR_RGB2BGR)
                #im_arr = tf.image.decode_jpeg(img.image, channels=3)
                #im_arr = im_arr.numpy()
                #im_arr = im_arr[:][top_crop:][:]
                #im_arr = im_arr[:][:-bot_crop][:]
                #im_data = Image.fromarray(im_arr)
                img_filename = '{0:07d}.png'.format(frame_idx)
                out_file = os.path.join(img_savepath, img_filename)
                plt.imsave(out_file, im_arr, format='png')
                #draw = ImageDraw.Draw(im_data)
                #im_data.save(out_file,'PNG')
    else:
        points_top_filtered = None

    pc_bin = points_top_filtered
    source_img = Image.open(out_file)
    draw = ImageDraw.Draw(source_img)
    json_calib = {}
    #print(frame.context)
    for calib in frame.context.laser_calibrations:
        if(laser_enum(calib.name)  == laser_enum.TOP):
            json_calib['beam_inclinations']   = []
            json_calib['beam_inclination_max'] = calib.beam_inclination_max
            json_calib['beam_inclination_min'] = calib.beam_inclination_min
            json_calib['extrinsic_transform'] = []
            for val in calib.beam_inclinations:
                json_calib['beam_inclinations'].append(val)
            ext = calib.extrinsic
            for val in ext.transform:
                json_calib['extrinsic_transform'].append(val)
            #print('found calibration')
            #print(calib)
    for calib in frame.context.camera_calibrations:
        if(cam_enum(calib.name)  == cam_enum.FRONT):
            json_calib['cam_intrinsic']           = []
            json_calib['cam_extrinsic_transform'] = []
            for val in calib.intrinsic:
                json_calib['cam_intrinsic'].append(val)
            for val in calib.extrinsic.transform:
                json_calib['cam_extrinsic_transform'].append(val)
    #TODO: Output calibration to JSON array file.
    k_l = 0
    json_labels                = {}
    json_labels['box']         = []
    json_labels['box_2d']      = []
    json_labels['class']       = []
    json_labels['meta']        = []
    json_labels['difficulty']  = []
    json_labels['id']          = []
    json_labels['assoc_frame'] = '{0:07d}'.format(i*1000+j) 
    json_labels['scene_name']  = frame.context.name
    json_labels['scene_type']  = []
    json_labels['scene_type'].append({'weather': frame.context.stats.weather,
                                        'tod': frame.context.stats.time_of_day})
    #print(json_calib)
    json_labels['calibration'] = []
    json_labels['calibration'].append(json_calib)
    for t, label in enumerate(frame.laser_labels):
        difficulty_override = 0
        if(label.num_lidar_points_in_box < 1):
            continue
        elif(label.num_lidar_points_in_box < 5):
            difficulty_override = 2

        #point 1 is near(x) left(y) bottom(z)
        #length: dim x
        x_c = float(label.box.center_x)
        #width: dim y
        y_c = float(label.box.center_y)
        #height: dim z
        z_c = float(label.box.center_z)
        #Second point is far(x) right(y) top(z)
        lx = float(label.box.length)
        wy = float(label.box.width)
        hz = float(label.box.height)
        delta_x = float(label.box.length)/2.0
        delta_y = float(label.box.width)/2.0
        delta_z = float(label.box.height)/2.0
        heading = float(label.box.heading)
        #Pointcloud to be cropped at x=[-40,40] y=[0,70] z=[0,10] as per config
        if(x_c < cfg.LIDAR.X_RANGE[0] or x_c > cfg.LIDAR.X_RANGE[1]):
            #print('object not infront of car')
            continue
        if(y_c < cfg.LIDAR.Y_RANGE[0] or y_c > cfg.LIDAR.Y_RANGE[1]):
            #print('object too far to left/right side')
            continue
        if(z_c < cfg.LIDAR.Z_RANGE[0] or z_c > cfg.LIDAR.Z_RANGE[1]):
            #print('object either too high or below car')
            continue

        #if(not skip_binaries):
        #    bbox = np.asarray([[z_c,y_c,z_c,lx,wy,hz,heading]])
        #    points_in_bbox = pc_points_in_bbox(pc_bin,bbox)
            #print(points_in_bbox.shape[0])
        #if(points_in_bbox.shape[0] < 5):
        #    continue
        #if(y2-y1 <= bbox_top_min and y1 == 0):
        #    continue

        # ANDREWS WORK
       

        bbox = np.zeros(7)       
        bbox[0] = x_c
        bbox[1] = y_c
        bbox[2] = z_c
        bbox[3] = lx
        bbox[4] = wy
        bbox[5] = hz
        bbox[6] = heading
        #bboxes.append(bbox)
        if(x_c <= 10):
            continue
        # x_c < 30m: min max transformed bbox points
        # x_c >= 30m: transformed lidar bboxes
        if(x_c < lidar_thresh_dist):
            pc_in_bbox = pc_points_in_bbox(points_top_filtered,bbox)
            if pc_in_bbox.shape[1] > 1:  # need 2 points for a bbox
                pc2_in_bbox = pc_points_in_bbox(points_top_filtered_2,bbox)  # second return
                transformed_pc = points_3D_to_image(json_calib, label.metadata, pc_in_bbox)
                #for point in transformed_pc:
                #    draw.point(point)
                bbox2D = transformed_pc_to_bbox(transformed_pc)      
        else:
            continue
        #else:
        #    pc_in_bbox = pc_points_in_bbox(points_top_filtered,bbox)  # need for intensity, elongation
        #    pc2_in_bbox = pc_points_in_bbox(points_top_filtered_2,bbox)
        #    bbox2D = label_3D_to_image(json_calib, label.metadata, label.box)  
        #    if(bbox2D is None):
        #        continue
        #    bbox2D = compute_2d_bounding_box(bbox2D)
    
        bbox2D_clipped = clip_2d_bounding_box(im_arr, bbox2D)
        truncation     = compute_truncation(bbox2D,bbox2D_clipped)
        draw.rectangle(bbox2D_clipped)
        #TODO: Need all points in the bbox to compute this!
        # Need avg intensity, elongation, return ratio
        if pc_in_bbox.shape[1]:
            avg_intensity  = np.mean(pc_in_bbox[:,:,3])
            avg_elongation = np.mean(pc_in_bbox[:,:,4])
        else:
            avg_intensity  = 0
            avg_elongation = 0
            return_ratio = 0
        if (pc2_in_bbox.shape[1] and pc_in_bbox.shape[1]):
            return_ratio   = pc2_in_bbox.shape[1]/pc_in_bbox.shape[1]  # divide num points from each return
        else:
            return_ratio   = 0

        json_labels['box'].append({
            'xc': '{:.3f}'.format(x_c),
            'yc': '{:.3f}'.format(y_c),
            'zc': '{:.3f}'.format(z_c),
            'lx': '{:.3f}'.format(lx),
            'wy': '{:.3f}'.format(wy),
            'hz': '{:.3f}'.format(hz),
            'heading': '{:.3f}'.format(heading),
        })
        #bbox_2d = frame.projected_lidar_labels[t]
        #x1 = bbox_2d.center_x - bbox_2d.width/2.0
        #x2 = bbox_2d.center_x + bbox_2d.width/2.0
        #y1 = bbox_2d.center_y - bbox_2d.length/2.0
        #y2 = bbox_2d.center_y + bbox_2d.length/2.0
        json_labels['box_2d'].append({
            'x1': '{:.3f}'.format(bbox2D_clipped[0]),
            'y1': '{:.3f}'.format(bbox2D_clipped[1]),
            'x2': '{:.3f}'.format(bbox2D_clipped[2]),
            'y2': '{:.3f}'.format(bbox2D_clipped[3])
        })
        json_labels['meta'].append({
            'vx':             '{:.3f}'.format(label.metadata.speed_x),
            'vy':             '{:.3f}'.format(label.metadata.speed_y),
            'ax':             '{:.3f}'.format(label.metadata.accel_x),
            'ay':             '{:.3f}'.format(label.metadata.accel_y),
            'pts':            '{:04d}'.format(label.num_lidar_points_in_box),
            'trunc':          '{:.2f}'.format(truncation),
            'avg_intensity':  '{:.2f}'.format(avg_intensity),
            'avg_elongation': '{:.2f}'.format(avg_elongation),
            'return_ratio':   '{:.2f}'.format(return_ratio)
        })
        json_labels['id'].append(label.id)
        json_labels['class'].append(label.type)
        if(difficulty_override != 0):
            json_labels['difficulty'].append(2)
        elif(label.detection_difficulty_level == 0):
            json_labels['difficulty'].append(1)
        else:
            json_labels['difficulty'].append(label.detection_difficulty_level)
        #print(json_labels)
        k_l = k_l + 1
    #print(k)
    k_i = 0
    source_img.save(out_file.replace('.png','_drawn.png'),'PNG')
    return json_labels


def filter_points(pc):
    pc_min_thresh = pc[(pc[:,0] >= cfg.LIDAR.X_RANGE[0]) & (pc[:,1] >= cfg.LIDAR.Y_RANGE[0]) & (pc[:,2] >= cfg.LIDAR.Z_RANGE[0])]
    pc_min_and_max_thresh = pc_min_thresh[(pc_min_thresh[:,0] < cfg.LIDAR.X_RANGE[1]) & (pc_min_thresh[:,1] < cfg.LIDAR.Y_RANGE[1]) & (pc_min_thresh[:,2] < cfg.LIDAR.Z_RANGE[1])]
    return pc_min_and_max_thresh


#COPIED FROM WAYMO DATASET, REMOVED CAMERA PROJECTION
def parse_range_image(frame,tfp):
  """Parse range images and camera projections given a frame.

  Args:
     frame: open dataset frame proto

  Returns:
     range_images: A dict of {laser_name,
       [range_image_first_return, range_image_second_return]}.
    range_image_top_pose: range image pixel pose for top lidar.
  """
  range_images = {}
  range_image_top_pose = None
  for laser in frame.lasers:
    if len(laser.ri_return1.range_image_compressed) > 0:  # pylint: disable=g-explicit-length-test
      range_image_str_tensor = tfp.io.decode_compressed(
          laser.ri_return1.range_image_compressed, 'ZLIB')
      ri = open_dataset.MatrixFloat()
      ri.ParseFromString(bytearray(range_image_str_tensor.numpy()))
      range_images[laser.name] = [ri]

      if laser.name == open_dataset.LaserName.TOP:
        range_image_top_pose_str_tensor = tfp.io.decode_compressed(
            laser.ri_return1.range_image_pose_compressed, 'ZLIB')
        range_image_top_pose = open_dataset.MatrixFloat()
        range_image_top_pose.ParseFromString(
            bytearray(range_image_top_pose_str_tensor.numpy()))

    if len(laser.ri_return2.range_image_compressed) > 0:  # pylint: disable=g-explicit-length-test
      range_image_str_tensor = tfp.io.decode_compressed(
          laser.ri_return2.range_image_compressed, 'ZLIB')
      ri = open_dataset.MatrixFloat()
      ri.ParseFromString(bytearray(range_image_str_tensor.numpy()))
      range_images[laser.name].append(ri)

  return range_images, range_image_top_pose


#COPIED FROM WAYMO DATASET, REMOVED CAMERA PROJECTION
def convert_range_image_to_point_cloud(frame,
                                       range_images,
                                       range_image_top_pose,
                                       tfp,
                                       ri_index=0):

  """Convert range images to point cloud.

  Args:
    frame: open dataset frame
     range_images: A dict of {laser_name, [range_image_first_return,
       range_image_second_return]}.
    range_image_top_pose: range image pixel pose for top lidar.
    ri_index: 0 for the first return, 1 for the second return.

  //   * channel 0: range
  //   * channel 1: intensity
  //   * channel 2: elongation
  //   * channel 3: is in any no label zone.

  Returns:
    points: {[N, 3]} list of 3d lidar points of length 5 (number of lidars).
    cp_points: {[N, 6]} list of camera projections of length 5
      (number of lidars).
  """
  from waymo_open_dataset.utils import transform_utils, range_image_utils
  calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
  points = []
  cp_points = []
  frame_pose = tfp.convert_to_tensor(
      value=np.reshape(np.array(frame.pose.transform), [4, 4]))
  # [H, W, 6]
  range_image_top_pose_tensor = tfp.reshape(
      tfp.convert_to_tensor(value=range_image_top_pose.data),
      range_image_top_pose.shape.dims)
  # [H, W, 3, 3]
  range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
      range_image_top_pose_tensor[..., 0], range_image_top_pose_tensor[..., 1],
      range_image_top_pose_tensor[..., 2])
  range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
  range_image_top_pose_tensor = transform_utils.get_transform(
      range_image_top_pose_tensor_rotation,
      range_image_top_pose_tensor_translation)
  for c in calibrations:
    range_image = range_images[c.name][ri_index]
    if len(c.beam_inclinations) == 0:  # pylint: disable=g-explicit-length-test
      beam_inclinations = range_image_utils.compute_inclination(
          tfp.constant([c.beam_inclination_min, c.beam_inclination_max]),
          height=range_image.shape.dims[0])
    else:
      beam_inclinations = tfp.constant(c.beam_inclinations)

    beam_inclinations = tfp.reverse(beam_inclinations, axis=[-1])
    extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

    range_image_tensor = tfp.reshape(
        tfp.convert_to_tensor(value=range_image.data), range_image.shape.dims)
    pixel_pose_local = None
    frame_pose_local = None
    if c.name == open_dataset.LaserName.TOP:
      pixel_pose_local = range_image_top_pose_tensor
      pixel_pose_local = tfp.expand_dims(pixel_pose_local, axis=0)
      frame_pose_local = tfp.expand_dims(frame_pose, axis=0)
    range_image_mask = range_image_tensor[..., 0] > 0
    range_image_cartesian = range_image_utils.extract_point_cloud_from_range_image(
        tfp.expand_dims(range_image_tensor[..., 0], axis=0),
        tfp.expand_dims(extrinsic, axis=0),
        tfp.expand_dims(tfp.convert_to_tensor(value=beam_inclinations), axis=0),
        pixel_pose=pixel_pose_local,
        frame_pose=frame_pose_local)

    range_image_cartesian = tfp.squeeze(range_image_cartesian, axis=0)
    points_tensor = tfp.gather_nd(range_image_cartesian,
                                  tfp.compat.v1.where(range_image_mask))
    intensity_tensor = tfp.gather_nd(tfp.expand_dims(range_image_tensor[..., 1], axis=2),
                                     tfp.compat.v1.where(range_image_mask)) 
    elongation_tensor = tfp.gather_nd(tfp.expand_dims(range_image_tensor[..., 2], axis=2),
                                      tfp.compat.v1.where(range_image_mask)) 
    stacked_tensor = np.hstack((points_tensor.numpy(),intensity_tensor.numpy(),elongation_tensor.numpy()))
    points.append(stacked_tensor)

  return points

def compute_2d_bounding_box(points):
    """Compute the 2D bounding box for a set of 2D points.
    
    img_or_shape: Either an image or the shape of an image.
                  img_or_shape is used to clamp the bounding box coordinates.
    
    points: The set of 2D points to use
    """

    # Compute the 2D bounding box and draw a rectangle
    x1 = np.amin(points[...,0])
    x2 = np.amax(points[...,0])
    y1 = np.amin(points[...,1])
    y2 = np.amax(points[...,1])

    return (x1,y1,x2,y2)

def clip_2d_bounding_box(img, points):
    shape = img.shape
    x1 = min(max(0,points[0]),shape[1])
    y1 = min(max(0,points[1]),shape[0])
    x2 = min(max(0,points[2]),shape[1])
    y2 = min(max(0,points[3]),shape[0])
    return (x1,y1,x2,y2)

def compute_truncation(points, clipped_points):

    clipped_area = (clipped_points[2] - clipped_points[0])*(clipped_points[3] - clipped_points[1])
    orig_area    = (points[2] - points[0])*(points[3] - points[1])
    if(clipped_area <= 0):
        return 1.0
    else:
        return clipped_area/orig_area

def points_3D_to_image(json_calib, metadata, points):  
    instrinsic = json_calib['cam_intrinsic']
    extrinsic = np.array(json_calib['cam_extrinsic_transform']).reshape(4,4)
    vehicle_to_image = get_image_transform(instrinsic, extrinsic)  # magic array 4,4 to multiply and get image domain

    points_transform = []
    for box_points in points:
        transformed_points = []
        point_xyz = box_points[:,0:3]
        one_concat = np.ones(len(point_xyz)) # ones to append
        one_concat = one_concat[:,np.newaxis]
        point_ext = np.concatenate((point_xyz,one_concat),axis=1)
        for point in point_ext:
            transformed_points.append(np.matmul(vehicle_to_image,point))
        transformed_points = np.asarray(transformed_points)
        points_transform.append(transformed_points)

    
    points_transform = np.asarray(points_transform) 
    for bbox_points in points_transform:
        if (not len(bbox_points)):  # prevent index error for 0 point bboxes
            continue
        bbox_points[:,0] = bbox_points[:,0]/bbox_points[:,2]  # x/z
        bbox_points[:,1] = bbox_points[:,1]/bbox_points[:,2]  # y/z
    
    return points_transform

def transformed_pc_to_bbox(points,draw=None):
    # input: point cloud in image domain, Can also draw rect on img
    # output: [x1,y1,x2,y2] bbox 
    
    if (not len(points)):  # no points
        return
    # topleft = (np.amin(points[:,0]),np.amin(points[:,1]))
    # botright = (np.amax(points[:,0]),np.amax(points[:,1]))
    coordinates = (np.amin(points[:,0]),np.amin(points[:,1]),np.amax(points[:,0]),np.amax(points[:,1]))
    if draw is not None:
        draw.rectangle(coordinates,outline=(255,0,0))
    return coordinates

def label_3D_to_image(json_calib, metadata, bbox):
    bbox_transform_matrix = get_box_transformation_matrix(bbox)  
    instrinsic = json_calib['cam_intrinsic']
    extrinsic = np.array(json_calib['cam_extrinsic_transform']).reshape(4,4)
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

def pc_points_in_bbox(point,box,draw=None):
    """Obtains point clouds inside bounding bboxes
       Currently used per box 
    Args:
        point: [N, 5] tensor. [x,y,z,intensity,elongation]
        box: [M, 7] tensor. Inner dims are: [center_x, center_y, center_z, length,
        width, height, heading].
        Returns:
        point_cloud: [N,5] np matrix. Cloud inside bbox
  """

    #with tf.compat.v1.name_scope(name, 'ComputeNumPointsInBox3D', [point, box]):
    # [N, M]
    point = tf.convert_to_tensor(point[:,:],dtype=tf.float32)
    box = tf.convert_to_tensor(box,dtype=tf.float32)
    point_xyz = point[:, 0:3]

    # Check if multiple or one bbox(7 values)
    if (box.get_shape() == 7):
        box = box[np.newaxis,:]

    point_in_box = tf.cast(is_within_box_3d(point_xyz, box), dtype=tf.int32)  # returns boolean, cast to int
    point_in_box = tf.transpose(point_in_box)
    point_in_box = np.asarray(point_in_box)
    point = np.asarray(point)
    points_vector = np.asarray(tf.reduce_sum(input_tensor=point_in_box, axis=1))
    num_points_in_box = tf.reduce_sum(input_tensor=point_in_box, axis=0)
    point_clouds = []
    for row in point_in_box:
        values = []
        idx = np.nonzero(row)  # grab indexs where there are values
        values.append(point[idx])  # fancy indexing
        values = np.asarray(values)
        values = np.reshape(values,(-1, 5))  # x,y,z,intensity,elongation
        point_clouds.append(values)
    #draw_points_in_bbox(point,points_draw_vector,draw)
    point_clouds = np.asarray(point_clouds)
    return point_clouds

def is_within_box_3d(point, box, name=None):
    center = box[:, 0:3]  # xc,yc,zc
    dim = box[:, 3:6] # L,W,H
    heading = box[:, 6]
    # [M, 3, 3]
    rotation = transform_utils.get_yaw_rotation(heading)  # rotation matrix 
    # [M, 4, 4]
    transform = transform_utils.get_transform(rotation, center)  # transform matrix
    # [M, 4, 4]
    transform = tf.linalg.inv(transform)
    # [M, 3, 3]
    rotation = transform[:, 0:3, 0:3]
    # [M, 3]
    translation = transform[:, 0:3, 3]  # translation matrix 

    # [N, M, 3]
    point_in_box_frame = tf.einsum('nj,mij->nmi', point, rotation) + translation  # 
    # [N, M, 3]
    point_in_box = tf.logical_and(point_in_box_frame <= dim * 0.5, point_in_box_frame >= -dim * 0.5)
    # [N, M]
    point_in_box = tf.cast(tf.reduce_prod(input_tensor=tf.cast(point_in_box, dtype=tf.uint8), axis=-1),dtype=tf.bool)
    return point_in_box 

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

def draw_transformed_pc(points, draw):
    # input multiple transformed point clouds 
    # each point cloud has x,y values to be drawn 

    # Draw all points
    for box in points:
        if (not len(box)):  # no points
            continue
        for point in box:
            xy = (point[0],point[1])
            draw.point(xy,fill=(255,0,0))

#Cheat to have secondary functions below main
if __name__ == '__main__':
    main()