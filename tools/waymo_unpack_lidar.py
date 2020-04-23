import _init_paths
import tensorflow as tf
tf.enable_eager_execution()
import os
import math
import numpy as np
import itertools
import json
from PIL import Image, ImageDraw
from enum import Enum
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset 
from multiprocessing import Process, Pool
import multiprocessing.managers
from concurrent.futures import ProcessPoolExecutor, as_completed
from model.config import cfg
import utils.bbox as bbox_utils

skip_binaries = False
class laser_enum(Enum):
    UNKNOWN     = 0
    TOP         = 1
    FRONT       = 2
    SIDE_LEFT   = 3
    SIDE_RIGHT  = 4
    REAR        = 5
def main():
    mypath = '/home/mat/thesis/data2/waymo/val'
    tfrecord_path = mypath + '/compressed_tfrecords'
    num_proc = 16
    #top_crop = 550
    bbox_top_min = 30
    file_list = [os.path.join(tfrecord_path,f) for f in os.listdir(tfrecord_path) if os.path.isfile(os.path.join(tfrecord_path,f))]
    file_list = sorted(file_list)
    #filename = 'segment-11799592541704458019_9828_750_9848_750_with_camera_labels.tfrecord'
    #pool = Pool(processes=16)
    #m = 
    with open(os.path.join(mypath,'labels','lidar_labels_new.json'), 'w') as json_file:
        json_struct = []
        for i,filename in enumerate(file_list):
            #if(i > 500):
            #    break
            if('tfrecord' in filename):
                print('opening {}'.format(filename))
                dataset = tf.data.TFRecordDataset(filename,compression_type='')
                dataset_list = []
                for elem in dataset:
                    dataset_list.append(elem)
                dataset_len = len(dataset_list)
                for j in range(0,dataset_len):
                    if(j%6 == 0):
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
    if(len(frame.no_label_zones) != 0):
        print('found a NLZ')
    if(not skip_binaries):
        (range_images, range_image_top_pose) = parse_range_image(frame,tfp)
        points     = convert_range_image_to_point_cloud(frame,range_images,range_image_top_pose, tfp)
        #points_2   = convert_range_image_to_point_cloud(frame,range_images,range_image_top_pose, tfp, ri_index=1)
        #Top extraction
        points_top       = points[laser_enum.TOP.value-1]
        #cp_points_top    = cp_points[laser_enum.TOP.value-1]
        #points_top_2     = points_2[laser_enum.TOP.value-1]
        points_top_filtered = filter_points(points_top)
        #cp_points_top_2  = cp_points_2[laser_enum.TOP.value-1]
        bin_filename = '{0:07d}.npy'.format(i*1000+j)
        out_file = os.path.join(mypath, 'point_clouds_new',bin_filename)
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
    else:
        points_top_filtered = None

    pc_bin = points_top_filtered

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
    #TODO: Output calibration to JSON array file.
    k_l = 0
    json_labels                = {}
    json_labels['box']         = []
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
    for label in frame.laser_labels:
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
        json_labels['box'].append({
            'xc': '{:.3f}'.format(x_c),
            'yc': '{:.3f}'.format(y_c),
            'zc': '{:.3f}'.format(z_c),
            'lx': '{:.3f}'.format(lx),
            'wy': '{:.3f}'.format(wy),
            'hz': '{:.3f}'.format(hz),
            'heading': '{:.3f}'.format(heading),
        })
        json_labels['meta'].append({
            'vx': '{:.3f}'.format(label.metadata.speed_x),
            'vy': '{:.3f}'.format(label.metadata.speed_y),
            'ax': '{:.3f}'.format(label.metadata.accel_x),
            'ay': '{:.3f}'.format(label.metadata.accel_y),
            'pts': '{}'.format(label.num_lidar_points_in_box)
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
    return json_labels


def pc_points_in_bbox(pc,bbox):
    #z_min = bbox[0][2] - bbox[0][5]/2.0
    #z_max = bbox[0][2] + bbox[0][5]/2.0
    bev_bboxes = bbox_utils.bbaa_graphics_gems(bbox,None,None,False)
    bev_bbox   = bev_bboxes[0]
    pc_min_thresh = pc[(pc[:,0] >= bev_bbox[0]) & (pc[:,1] >= bev_bbox[1]) & (pc[:,2] >= cfg.LIDAR.Z_RANGE[0])]
    pc_min_and_max_thresh = pc_min_thresh[(pc_min_thresh[:,0] <= bev_bbox[2]) & (pc_min_thresh[:,1] <= bev_bbox[3]) & (pc_min_thresh[:,2] <= cfg.LIDAR.Z_RANGE[1])]
    return pc_min_and_max_thresh


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




#Cheat to have secondary functions below main
if __name__ == '__main__':
    main()
