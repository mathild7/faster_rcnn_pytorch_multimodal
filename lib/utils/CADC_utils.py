""" Helper methods for loading and parsing KITTI data.
Author: Charles R. Qi, Kui Xu
Date: September 2017/2018
"""
from __future__ import print_function

import numpy as np
import cv2
import os,math
from scipy.optimize import leastsq
from PIL import Image

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
    pts_2d = np.zeros((points_exp.shape[0],3))
    for i, point in enumerate(points_exp):
        pts_2d[i] = np.matmul(transform_matrix,point)[0:3]
    #projected_pts = np.einsum("bij, bjk -> bik", batch_transform_matrix, points_exp)[:,:,0]
    pts_2d[:,0] /= pts_2d[:,2]
    pts_2d[:,1] /= pts_2d[:,2]
    return pts_2d[:,0:2]
