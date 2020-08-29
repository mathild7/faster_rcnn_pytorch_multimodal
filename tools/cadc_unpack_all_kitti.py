#load calibration.py has all the calbration matrices

#!/usr/bin/env python

'''
want only annotations that have at least 20 lidar points
writes out annotation files
'''
import sys
#print(sys.path)

import json
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import os.path
from shutil import copyfile
import yaml
import math

base_dir        = '/home/mat/thesis/data/cadc/cadcd'
drive_dir       = ['2018_03_06','2018_03_07','2019_02_27']
#2018_03_06
# Seq | Snow  | Road Cover | Lens Cover
#   1 | None  |     N      |     N
#   5 | Med   |     N      |     Y
#   6 | Heavy |     N      |     Y
#   9 | Light |     N      |     Y
#  18 | Light |     N      |     N

#2018_03_07
# Seq | Snow  | Road Cover | Lens Cover
#   1 | Heavy |     N      |     Y
#   4 | Light |     N      |     N
#   5 | Light |     N      |     Y

#2019_02_27
# Seq | Snow  | Road Cover | Lens Cover
#   5 | Light |     Y      |     N
#   6 | Heavy |     Y      |     N
#  15 | Med   |     Y      |     N
#  28 | Light |     Y      |     N
#  37 | Extr  |     Y      |     N
#  46 | Extr  |     Y      |     N
#  59 | Med   |     Y      |     N
#  73 | Light |     Y      |     N
#  75 | Med   |     Y      |     N
#  80 | Heavy |     Y      |     N

#val_seq_sel     = [[1,5,6,18],[1,4,5],[5,6,15,28,37,46,59,73,75,80]]
#Heavy snow
val_seq_sel     = [[],[],[11,20,41,43,46,49,51,65,68,70,78]]
#Partial camera coverage
#val_seq_sel     = [[5,6,8,9,10],[1,2,5,6,7],[]]
train_ratio     = 0.80
MIN_NUM_POINTS  = 5
crop_top        = 150
crop_bottom     = 250
def load_calibration(calib_path):
  calib = {}

  # Get calibrations
  calib['extrinsics'] = yaml.load(open(calib_path + '/extrinsics.yaml'), yaml.SafeLoader)
  calib['CAM00'] = yaml.load(open(calib_path + '/00.yaml'), yaml.SafeLoader)
  #calib['CAM01'] = yaml.load(open(calib_path + '/01.yaml'), yaml.SafeLoader)
  #calib['CAM02'] = yaml.load(open(calib_path + '/02.yaml'), yaml.SafeLoader)
  #calib['CAM03'] = yaml.load(open(calib_path + '/03.yaml'), yaml.SafeLoader)
  #calib['CAM04'] = yaml.load(open(calib_path + '/04.yaml'), yaml.SafeLoader)
  #calib['CAM05'] = yaml.load(open(calib_path + '/05.yaml'), yaml.SafeLoader)
  #calib['CAM06'] = yaml.load(open(calib_path + '/06.yaml'), yaml.SafeLoader)
  #calib['CAM07'] = yaml.load(open(calib_path + '/07.yaml'), yaml.SafeLoader)

  return calib

#TODO: Save R0_rect, T_cam_to_velo, P0 and D0
def make_calib_file(calib, calib_target_path):
    f_c = open(calib_target_path, "w+")
    for k,v in calib.items():
        if(k == 'extrinsics'):
            for k_e,v_e in v.items():
                f_c.write(k_e)
                f_c.write(': ')
                if(type(v_e) is list):
                    for value in v_e:
                        if(type(value) is list):
                            for value2 in value:
                                f_c.write(str(value2))
                                f_c.write(' ')
                        else:
                            f_c.write(str(value))
                            f_c.write(' ')
                else:
                    f_c.write(str(v_e))
                    f_c.write(' ')
                f_c.write('\n')
        elif('CAM' in k and type(v) is dict):
            f_c.write('{:s}_matrix'.format(k))
            f_c.write(': ')
            for value in v['camera_matrix']['data']:
                if(type(value) is list):
                    for value2 in value:
                        f_c.write(str(value2))
                        f_c.write(' ')
                else:
                    f_c.write(str(value))
                    f_c.write(' ')
            f_c.write('\n')
            f_c.write('{:s}_distortion'.format(k))
            f_c.write(': ')
            for value in v['distortion_coefficients']['data']:
                if(type(value) is list):
                    for value2 in value:
                        f_c.write(str(value2))
                        f_c.write(' ')
                else:
                    f_c.write(str(value))
                    f_c.write(' ')
        else:
            f_c.write(k)
            f_c.write(': ')
            if(type(v) is list):
                for value in v:
                    if(type(value) is list):
                        for value2 in value:
                            f_c.write(str(value2))
                            f_c.write(' ')
                    else:
                        f_c.write(str(value))
                        f_c.write(' ')
            else:
                f_c.write(str(v))
                f_c.write(' ')
            f_c.write('\n')
    f_c.close()

#Create target Directory if don't exist
def create_output_directory(camera):
    home_dir = os.path.join(base_dir,'processed')
    modes = ['train','val']
    for mode in modes:
        #current_dir ="%s/annotation" %pwd
        dir_names = []
        dir_names.append('image_0{:1d}'.format(int(camera)))
        dir_names.append('point_clouds')
        dir_names.append('calib')
        dir_names.append("annotation_0{:01d}".format((int(camera))))
        for dir_name in dir_names:
            target_path = os.path.join(home_dir,mode,dir_name)
            if not os.path.exists(target_path):
                os.makedirs(target_path)
                print("Directory ", target_path,  " Created ")
            else:
                print("Directory ", target_path,  " already exists")
    return home_dir

def write_txt_annotation(frame,cam,out_dir,mode,image_dir,drive,seq,drive_and_seq):
    cam = str(cam)
    DISTORTED = False

    writeout = True
    filter = MIN_NUM_POINTS  #only show/write out annotations that have at least 10 lidar points in them

    img_path = image_dir + '/' + format(frame, '010') + ".png"
    print(img_path)
    if DISTORTED:
      path_type = 'raw'
    else:
      path_type = 'processedmoose'
    annotations_file = os.path.join(base_dir,drive,seq,'3d_ann.json') #contains all the annotations
    lidar_path = os.path.join(base_dir, drive,seq, "labeled", "lidar_points", "data", format(frame, '010') + ".bin")  #change this to snowy or not snowy
    calib_path = os.path.join(base_dir, drive, 'calib')
    calib = load_calibration(calib_path)


    # Load 3d annotations
    annotations_data = None
    with open(annotations_file) as f:
        annotations_data = json.load(f)


    cam_intrinsic = np.eye(4)  #identity matrix
    cam_intrinsic[0:3,0:3] = np.array(calib['CAM0' + cam]['camera_matrix']['data']).reshape(-1, 3)  #K_xx camera projection matrix (3x3)
    cam_extrinsic = np.array(calib['extrinsics']['T_LIDAR_CAM0' + cam])  #
    cam_distorted = calib['CAM0' + cam]['distortion_coefficients']['data']  #D_xx camera distortion matrix
    k1 = cam_distorted[0]
    k2 = cam_distorted[1]
    p1 = cam_distorted[2]
    p2 = cam_distorted[3]
    k3 = cam_distorted[4]

    fc1 = cam_intrinsic[0][0]
    fc1_alpha_c = cam_intrinsic[0][1]
    cc1 = cam_intrinsic[0][2]
    fc2 = cam_intrinsic[1][1]
    cc2 = cam_intrinsic[1][2]

    # Projection matrix from camera to image frame
    #T_IMG_CAM = np.eye(4); #identity matrix
    #T_IMG_CAM[0:3,0:3] = np.array(calib['CAM0' + cam]['camera_matrix']['data']).reshape(-1, 3); #camera to image #this comes from e.g "F.yaml" file

    #T_IMG_CAM : 4 x 4 matrix
    #T_IMG_CAM = T_IMG_CAM[0:3,0:4]; # remove last row, #choose the first 3 rows and get rid of the last row

    #T_CAM_LIDAR = np.linalg.inv(np.array(calib['extrinsics']['T_LIDAR_CAM0' + cam])); #going from lidar to camera

    #T_IMG_LIDAR = np.matmul(T_IMG_CAM, T_CAM_LIDAR); # go from lidar to image

    img = cv2.imread(img_path)
    img_h, img_w = img.shape[:2]  #get the image height and width
    img = img[crop_top:,:,:]
    img = img[:-crop_bottom,:,:]

    #Add each cuboid to image
    #the coordinates in the tracklet json are lidar coords

    #drive_and_seq = int(seq)*1000 + drive_dir.index(drive)*100000
    #if cam =='0':
    #    cam_id   = 0
    #elif cam =='1':
    #    cam_id   = 1*10000000
    #elif cam == '2':
    #    cam_id = 2*10000000
    #elif cam == '3':
    #    cam_id = 3*10000000
    #elif cam =='4':
    #    cam_id = 4*10000000
    #elif cam =='5':
    #    cam_id = 5*10000000
    #elif cam =='6':
    #    cam_id = 6*10000000
    #else:
    #    cam_id = 7*10000000
    frame_id     = frame + drive_and_seq
    img_file     = os.path.join(out_dir,mode,'image_0{:01d}'.format(int(cam)),'{:07d}.png'.format(frame_id))
    target_lidar = os.path.join(out_dir,mode,'point_clouds','{:07d}.bin'.format(frame_id))
    annotation_target_path = os.path.join(out_dir,mode,'annotation_0{:01d}'.format(int(cam)),'{:07d}.txt'.format(frame_id))
    calib_target_path      = os.path.join(out_dir,mode,'calib','{:07d}.txt'.format(frame_id))
    f = open(annotation_target_path, "w+")
    if(not os.path.isfile(calib_target_path)):
        make_calib_file(calib, calib_target_path)
    for cuboid in annotations_data[frame]['cuboids']:


        #T_Lidar_Cuboid = np.eye(4); #identity matrix
        #T_Lidar_Cuboid[0:3,0:3] = R.from_euler('z', cuboid['yaw'], degrees=False).as_dcm(); #make a directional cosine matrix using the yaw, i.e rotation about z, yaw angle

        '''
        Rotations in 3 dimensions can be represented by a sequece of 3 rotations around a sequence of axes. 
        In theory, any three axes spanning the 3D Euclidean space are enough. In practice the axes of rotation are chosen to be the basis vectors.

        The three rotations can either be in a global frame of reference (extrinsic) or in a body centred frame of refernce (intrinsic),
        which is attached to, and moves with, the object under rotation
        
        In our case, 'z' specifies the axis of rotation (extrinsic rotation), cudoid['yaw] euler angles in radians (rotation angle)
        Returns object containing the rotation represented by the sequencce of rotations around given axes with given angles
        
        T_Lidar_Cuboid = basic rotation (elemental rotation) : R_z(theta = yaw) = [[ cos(theta) - sin(theta) 0 etc]]
        * .as_dcm - Represent as direction cosine matrices.

        3D rotations can be represented using direction cosine matrices, which are 3 x 3 real orthogonal matrices with determinant equal to +1
        
        '''
        num_lidar_points     = cuboid['points_count']

        #the above is the translation part of the transformation matrix, so now we have the rotation and the translation

        length = cuboid['dimensions']['y'] #x is just a naming convention that scale uses, could have easily been cuboid['dimensions']['width']
        width = cuboid['dimensions']['x']
        height = cuboid['dimensions']['z']
        bbox_3d = [cuboid['position']['x'],cuboid['position']['y'],cuboid['position']['z'],length,width,height,cuboid['yaw']]
        #radius = 3
        bbox_transform_matrix = get_box_transformation_matrix(bbox_3d)
        bbox_img = np.matmul(np.linalg.inv(cam_extrinsic),bbox_transform_matrix)
        bbox_img_yaw = -cuboid['yaw'] + np.pi/2.0
        #if bbox_img_yaw < -np.pi:
        #    rotation_y = bbox_img_yaw + 2*np.pi
        #elif bbox_img_yaw  > np.pi:
        #    rotation_y = bbox_img_yaw - 2 * np.pi

        #else:
        rotation_y = bbox_img_yaw
        bbox_3d_cam = [bbox_img[0][3],bbox_img[1][3],bbox_img[2][3],length,width,height,rotation_y]
        #note that in the lidar frame, up is z, forwards is x, side is y
        if(cuboid['position']['x'] - length/2.0 <= 0):
            continue
        '''
        Very import to remember that The LiDAR frame has x forward, y left and z up. The given yaw is around the z axis
        the cuboid frame which we create has the same axis
        '''
        scan_data = np.fromfile(lidar_path, dtype=np.float32)  # numpy from file reads binary file
        lidar = scan_data.reshape((-1, 4))

        [rows,cols] = lidar.shape  # rows and cols of the lidar points, rows = number of lidar points, columns (4 = x , y, z , intensity)

        # 0: front , 1: front right, 2: right front, 3: back right, 4: back, 5: left back, 6: left front, 7: front left

        if num_lidar_points > filter: #if we filter such that the annotations should have at least 20 lidar
            lidar_to_image = get_image_transform(cam_intrinsic, cam_extrinsic)  # magic array 4,4 to multiply and get image domain
            #bbox_transform_matrix = transform_axes(bbox_transform_matrix)
            box_to_image = np.matmul(lidar_to_image, bbox_transform_matrix)
            vertices = np.empty([2,2,2,2])
            ignore = False
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
                            ignore = True
                        
                        x = v[0]/v[2]
                        y = v[1]/v[2]
                        #r = np.sqrt(x**2 + y**2)
                        #radial_correction = (1+k1 * r**2 + k2 * r**4 + k3 * r**6)
                        #x_tang_correction = 2*p1 * x * y + p2 * (r**2 + 2 * x**2)
                        #y_tang_correction = p1 * (r**2 + 2 * y**2) * x * y + 2 * p2 * (x * y)
                        #x_corr = x*radial_correction + x_tang_correction
                        #y_corr = y*radial_correction + y_tang_correction
                        #x_p    = fc1*x_corr + fc1_alpha_c*y_corr + cc1
                        #y_p    = fc2*y_corr + cc2
                        vertices[k,l,m,:] = [x, y]

            vertices = vertices.astype(np.int32)
            vert_2d  = compute_2d_bounding_box(img,vertices)
            x1,y1,x2,y2 = vert_2d
            if(ignore):
                continue


            #print("This is the y rotation")
            #print(rotation_y/np.pi * 180)

            #print("This is the given yaw in lidar frame")
            #print (cuboid['yaw']/np.pi*180)
            viewing_angle = np.arctan2(bbox_3d_cam[0], bbox_3d_cam[2]) #arctan2(x/z)

            #print("This is the viewing angle")
            #print(viewing_angle/np.pi * 180)

            alpha_tmp = rotation_y - viewing_angle

            #if alpha_tmp < -np.pi:
            #    alpha = alpha_tmp + 2*np.pi

            #elif alpha_tmp > np.pi:
            #    alpha = alpha_tmp - 2*np.pi

            #else:
            alpha = alpha_tmp

            #print("This is alpha")
            #print(alpha/np.pi*180)

            #print("***************************")



            #truncation calculation

            x_min_set = min(img_w-1,max(0,x1))
            y_min_set = min(img_h-1,max(0,y1)) - crop_top
            x_max_set = min(img_w-1,max(0,x2))
            y_max_set = min(img_h-1,max(0,y2)) - crop_top


            area_actual = (y1 - x1) * (y2 - y1)
            area_set = (x_max_set - x_min_set) * (y_max_set - y_min_set)

            if (x_min_set < 0 and x_max_set > img_w):  #get rid of any weird huge bb, where x min and x max span over the whole image
                continue

            if (y_min_set < 0 and y_max_set > img_h):  #get rid of any huge bb where y min and y max span over the whole image
                continue

            if area_set == 0:  #tracklet is outside of the image
                continue
            if(area_actual <= 0):
                ratio_of_area = 0
            else:
                ratio_of_area = area_set/area_actual

            if ratio_of_area == 1:
                truncation = 0

            else:
                truncation = 1 - ratio_of_area


            '''
            example of a cuboid statement: {'uuid': '33babea4-958b-49a1-ac65-86174faa111a', 'attributes': {'state': 'Moving'}, 'dimensions': {'y': 4.276, 'x': 1.766, 'z': 1.503}, 'label': 'Car', 'position': {'y': 5.739311373648604, 'x': 57.374972338211876, 'z': -1.5275162154592332}, 'camera_used': None, 'yaw': -3.1134003618947323, 'stationary': False}
                '''
            #cv2.rectangle(img, (x_min_set, y_min_set), (x_max_set,y_max_set), (0,255,0),2)
            f.write(cuboid['label'])
            f.write(' %s '%(round(truncation,2)))  #trucation
            f.write('0 ')  #occlusion
            f.write('%s ' %(round(alpha,2)))
            f.write('{:.1f} {:.1f} {:.1f} {:.1f} '.format(x_min_set, y_min_set, x_max_set, y_max_set)) #pixel
            f.write('{:.3f} {:.3f} {:.3f} '.format(bbox_3d[3],bbox_3d[4],bbox_3d[5]))
            f.write('{:.3f} {:.3f} {:.3f} '.format(bbox_3d[0],bbox_3d[1],bbox_3d[2]))
            f.write('{:.5f} '.format(bbox_3d[6]))
            f.write('{:d} '.format(num_lidar_points))
            f.write('{:s} '.format(drive))
            f.write('{:s} '.format(seq))
            f.write('\n')
    f.close()
    copyfile(lidar_path,target_lidar)
    #pts = project_pts()
    #for point in pts:
    #    cv2.point(point)
    cv2.imwrite(img_file, img)
    #cv2.imshow('image',img)
    #cv2.waitKey(10000)
    #cv2.imwrite("image1.jpg")

def get_image_transform(intrinsic, extrinsic):
    """ For a given camera calibration, compute the transformation matrix
        from the vehicle reference frame to the image space.
    """
    # Camera model:
    # | fx  0 cx 0 |
    # |  0 fy cy 0 |
    # |  0  0  1 0 |

    # Swap the axes around
    # Compute the projection matrix from the vehicle space to image space.
    lidar_to_img = np.matmul(intrinsic, np.linalg.inv(extrinsic))
    return lidar_to_img
    #return np.linalg.inv(extrinsic)

def project_pts(calib_file,points):
    fd = open(calib_file,'r').read().splitlines()
    cam_intrinsic = np.eye(4)  #identity matrix
    for line in fd:
        matrix = line.rstrip().split(' ')
        if(matrix[0] == 'T_LIDAR_CAM00:'):
            cam_extrinsic = np.array(matrix[1:]).astype(np.float32)[np.newaxis,:].reshape(4,4)
        if(matrix[0] == 'CAM00_matrix:'):
            cam_intrinsic[0:3,0:3] = np.array(matrix[1:]).astype(np.float32).reshape(3, 3)  #K_xx camera projection matrix (3x3)
    transform_matrix = get_image_transform(cam_intrinsic, cam_extrinsic)  # magic array 4,4 to multiply and get image domain
    points_exp = np.ones((points.shape[0],4))
    points_exp[:,0:3] = points
    points_exp = points_exp[:,:]
    #batch_transform_matrix = np.repeat(transform_matrix[np.newaxis,:,:],points_exp.shape[0],axis=0)
    projected_points = np.zeros((points_exp.shape[0],3))
    for i, point in enumerate(points_exp):
        projected_points[i] = np.matmul(transform_matrix,point)[0:3]
    #projected_pts = np.einsum("bij, bjk -> bik", batch_transform_matrix, points_exp)[:,:,0]
    return projected_points

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

def get_box_transformation_matrix(box):
    """Create a transformation matrix for a given label box pose."""

    tx,ty,tz = box[0],box[1],box[2]
    c = math.cos(box[6])
    s = math.sin(box[6])

    sl, sh, sw = box[3], box[4], box[5]

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

    # Swap the axes around
    # Compute the projection matrix from the vehicle space to image space.
    lidar_to_img = np.matmul(intrinsic, np.linalg.inv(extrinsic))
    return lidar_to_img
    #return np.linalg.inv(extrinsic)

def transform_axes(bbox_transform_matrix):
    axes_transformation = np.array([
        [0,-1,0,0],
        [0,0,-1,0],
        [1,0,0,0],
        [0,0,0,1]])
    return np.matmul(axes_transformation,bbox_transform_matrix)

def write_all_annotations(cam):
    cam = str(cam)
    out_dir = create_output_directory(cam)
    for i,ddir in enumerate(drive_dir):
        data_dir = os.path.join(base_dir,ddir)
        sub_folders = sorted(os.listdir(data_dir))
        calib_idx = sub_folders.index('calib')
        del sub_folders[calib_idx]
        seq_count = len(sub_folders)

        for sequence in sub_folders:
            drive_and_sequence = int(sequence)*100 + drive_dir.index(ddir)*10000
            if(sequence == 'calib'):
                print('something very wrong has happened')
            #if(i > train_ratio*seq_count):
            #    mode = 'val'
            #else:
            #    mode = 'train'
            if(int(sequence) in val_seq_sel[i]):
                mode = 'val'
                print('val')
            else:
                print('train')
                mode = 'train'
            image_dir = os.path.join(data_dir,sequence, 'labeled','image_0' + cam, "data")
            seq_dir   = os.path.join(data_dir,sequence)
            file_names = sorted(os.listdir(image_dir))
            for frame in range(len(file_names)):
                write_txt_annotation(frame,cam,out_dir,mode,image_dir,ddir,sequence,drive_and_sequence)

if __name__ == '__main__':
    args = sys.argv
    if(len(args) > 1):
        if(args[1] == 'unpack'):
            base_dir = args[2]
            for cam in range(1):
                cam = str(cam)
                write_all_annotations(cam)
        elif('help' in args[1]):
            print('example usage:')
            print('    python3 filename.py unpack /path/to/download_data_dir ')
        else:
            print('unknown command, please type in --help as the argument')
    else:
        print('No input args specified, using default values. DANGEROUS!!')
        for cam in range(1):
            cam = str(cam)
            write_all_annotations(cam)
