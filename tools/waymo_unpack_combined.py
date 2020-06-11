import _init_paths
import tensorflow as tf
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
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

tf.enable_eager_execution()

top_crop = 300
bot_crop = 50
bbox_edge_thresh = 10
save_imgs = True
mypath = '/home/mat/thesis/data2/waymo/train'
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
    mypath = '/home/mat/thesis/data2/waymo/train'
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
    with open(os.path.join(mypath,'labels','lidar_labels_new.json'), 'w') as json_file:
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
        points_top_filtered_2 = points_top_2[laser_enum.TOP.value-1]
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
                im_arr = im_arr[:][top_crop:][:]
                im_arr = im_arr[:][:-bot_crop][:]
                #im_data = Image.fromarray(im_arr)
                img_filename = '{0:07d}.png'.format(frame_idx)
                out_file = os.path.join(img_savepath, img_filename)
                plt.imsave(out_file, im_arr, format='png')
                #draw = ImageDraw.Draw(im_data)
                #im_data.save(out_file,'PNG')
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

        bbox2D = label_3D_to_image(json_calib, label.metadata, label.box)  
        if(bbox2D is None):
            continue
        bbox2D         = compute_2d_bounding_box(bbox2D)
        bbox2D_clipped = compute_2d_bounding_box(im_arr, bbox2D)
        truncation     = compute_truncation(bbox2D,bbox2D_clipped)
        #TODO: Need all points in the bbox to compute this!
        avg_intensity  = 0
        avg_elongation = 0
        return_ratio   = 0
        #if(not skip_binaries):
        #    bbox = np.asarray([[z_c,y_c,z_c,lx,wy,hz,heading]])
        #    points_in_bbox = pc_points_in_bbox(pc_bin,bbox)
            #print(points_in_bbox.shape[0])
        #if(points_in_bbox.shape[0] < 5):
        #    continue
        #if(y2-y1 <= bbox_top_min and y1 == 0):
        #    continue

        #ANDREWS WORK

        json_labels['box'].append({
            'xc': '{:.3f}'.format(x_c),
            'yc': '{:.3f}'.format(y_c),
            'zc': '{:.3f}'.format(z_c),
            'lx': '{:.3f}'.format(lx),
            'wy': '{:.3f}'.format(wy),
            'hz': '{:.3f}'.format(hz),
            'heading': '{:.3f}'.format(heading),
        })
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


#Cheat to have secondary functions below main
if __name__ == '__main__':
    main()
