import tensorflow as tf
import _init_paths
import os
import math
import numpy as np
import itertools
import json
from PIL import Image, ImageDraw
from enum import Enum
from model.config import cfg
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

__all__ = ['is_within_box_3d', 'pc_points_in_bbox','get_upright_3d_box_corners', 'transform_point', 'transform_box']

# select all point cloud points contained within specific 3d bounding box
# elongation, intensity
def pc_points_in_bbox(point,box,draw,name=None):
    """Computes the number of points in each box given a set of points and boxes.
        TAKEN FROM WAYMO-GITHUB
    Args:
        point: [N, 3] tensor. Inner dims are: [x, y, z].
        box: [M, 7] tenosr. Inner dims are: [center_x, center_y, center_z, length,
        width, height, heading].
        name: tf name scope.
        Returns:
        num_points_in_box: [M] int32 tensor.
  """

    #with tf.compat.v1.name_scope(name, 'ComputeNumPointsInBox3D', [point, box]):
    # [N, M]
    point_xyz = point[:, 0:3]
    point_in_box = tf.cast(is_within_box_3d(point_xyz, box, name), dtype=tf.int32)  # returns boolean, cast to int
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

def draw_points_in_bbox(points,draw_vector,draw):
    for point in range(len(draw_vector)):
        if draw_vector[point]>0:
            z = int(((points[point,2]+3)/6) *255) # 0-6
            xy = (points[point,0]*10,(points[point,1]+40)*10)
            draw.point(xy,fill=(0,0,z))

def is_within_box_3d(point, box, name=None):
    #with tf.compat.v1.name_scope(name, 'IsWithinBox3D', [point, box]):
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

def label_3D_to_image_bbox(camera_calib, lidar_calib, metadata, bbox):
    points_transform_matrix = get_box_transformation_matrix(bbox)  
    instrinsic = camera_calib['intrinsic']
    extrinsic = np.array(camera_calib['extrinsic_transform']).reshape(4,4)
    vehicle_to_image = get_image_transform(instrinsic, extrinsic)  # magic array 4,4 to multiply and get image domain
    box_to_image = np.matmul(vehicle_to_image, points_transform_matrix)

    # Loop through the 8 corners constituting the 3D box
    # and project them onto the image
    vertices = np.empty([2,2,2,2])
    # 1: 000, 2: 001, 3: 010:, 4: 100k
    for k in [0, 1]:
        for l in [0, 1]:
            for m in [0, 1]:
                # 3D point in the box space
                v = np.array([(k-0.5), (l-0.5), (m-0.5), 1.])

                v = np.matmul(box_to_image, v)

                # If any of the corner is behind the camera, ignore this object.
                if v[2] < 0:
                    return None

                vertices[k,l,m,:] = [v[0]/v[2], v[1]/v[2]]   
  
    return vertices

def label_3D_to_image(camera_calib, lidar_calib, metadata, points):  
    instrinsic = camera_calib['intrinsic']
    extrinsic = np.array(camera_calib['extrinsic_transform']).reshape(4,4)
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

def filter_points(pc):
    pc_min_thresh = pc[(pc[:,0] >= cfg.LIDAR.X_RANGE[0]) & (pc[:,1] >= cfg.LIDAR.Y_RANGE[0]) & (pc[:,2] >= cfg.LIDAR.Z_RANGE[0])]
    pc_min_and_max_thresh = pc_min_thresh[(pc_min_thresh[:,0] < cfg.LIDAR.X_RANGE[1]+10) & (pc_min_thresh[:,1] < cfg.LIDAR.Y_RANGE[1]) & (pc_min_thresh[:,2] < cfg.LIDAR.Z_RANGE[1])]
    return pc_min_and_max_thresh

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
    
    # Draw bbox
    for box in points:
        if (not len(box)):  # no points
            continue
        botleft = (np.amin(box[:,0]),np.amin(box[:,1]))
        topright = (np.amax(box[:,0]),np.amax(box[:,1]))
        coordinates = (botleft,topright)
        draw.rectangle(coordinates,outline=(255,0,0))
        
        
save_imgs = True
mypath = '/home/mat/thesis/data2/waymo/val'
savepath = os.path.join(mypath,'images_new')
if not os.path.isdir(savepath):
    print('making path: {}'.format(savepath))
    os.makedirs(savepath)
tfrec_path = os.path.join(mypath,'compressed_tfrecords')
top_crop = 550
bbox_top_min = 30
file_list = [os.path.join(tfrec_path,f) for f in os.listdir(tfrec_path) if os.path.isfile(os.path.join(tfrec_path,f))]
file_list = sorted(file_list)
#filename = 'segment-11799592541704458019_9828_750_9848_750_with_camera_labels.tfrecord'
with open(os.path.join(mypath,'labels','image_labels_new.json'), 'w') as json_file:
    json_struct = []
    import tensorflow as tfp
    tfp.enable_eager_execution()
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
                    (range_images, range_image_top_pose) = parse_range_image(frame,tfp)  # 
                    points     = convert_range_image_to_point_cloud(frame,range_images,range_image_top_pose, tfp)  # all 5 lidars
                    #points_2   = convert_range_image_to_point_cloud(frame,range_images,range_image_top_pose, tfp, ri_index=1)
                    #Top extraction
                    points_top       = points[laser_enum.TOP.value-1]  # just the top lidar
                    #cp_points_top    = cp_points[laser_enum.TOP.value-1]
                    #points_top_2     = points_2[laser_enum.TOP.value-1]
                    points_top_filtered = filter_points(points_top)  # nX3 matrix 
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
                            out_file = os.path.join(savepath ,img_filename)
                            draw = ImageDraw.Draw(im_data)
                    assert im_data is not None
                    bboxes = []
                    for label in frame.laser_labels:                        
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
                        
                        #for finding pc_in_bbox
                        bbox = np.zeros(7)       
                        bbox[0] = x_c
                        bbox[1] = y_c
                        bbox[2] = z_c
                        bbox[3] = lx
                        bbox[4] = wy
                        bbox[5] = hz
                        bbox[6] = heading
                        bboxes.append(bbox)                     
                        
                        #Transform to image domain
                        bbox = label.box
                      
                        bbox2D = label_3D_to_image_bbox(img_calib, laser_calib, label.metadata, bbox)  
                        if(bbox2D is None):
                            continue
                        # bbox2D has format [x1,y1,x2,y2]
                        bbox2D = compute_2d_bounding_box(im_arr, bbox2D)
                        draw.rectangle(bbox2D)


                    bboxes = np.asarray(bboxes)
                    # Take points top filtered, obtain pc in bbox, tranform to image domain, draw 
                    points_top_filtered = tf.convert_to_tensor(points_top_filtered[:,:],dtype=tf.float32)
                    bboxes = tf.convert_to_tensor(bboxes,dtype=tf.float32)
                    pcs_in_bbox = pc_points_in_bbox(points_top_filtered,bboxes,draw)       

                    points_draw = label_3D_to_image(img_calib, laser_calib, label.metadata, pcs_in_bbox)      
                    draw_transformed_pc(points_draw,draw)              
                    im_data.save(out_file,'PNG')   
