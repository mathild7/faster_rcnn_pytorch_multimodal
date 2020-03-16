"""
Generates 3D anchors, placing them on the ground plane
STOLEN FROM AVOD: https://github.com/kujason/avod/blob/master/avod/core/anchor_generators/grid_anchor_3d_generator.py
"""

import numpy as np
import math
from model.config import cfg

class GridAnchor3dGenerator(object):

    def name_scope(self):
        return 'GridAnchor3dGenerator'

    def _generate(self):
        """
        Generates 3D anchors in a grid in the provided 3d area and places
        them on the ground_plane.

        Args:
            **params:
                area_3d: [[min_x, max_x], [min_y, max_y], [min_z, max_z]]
                This area describes the BEV in a 800x700x8 pixel image view.

        Returns:
            list of 3D anchors in the form N x [x, y, z, l, w, h, ry]
            These describe the space that RoI's are collected from.
        """
        x_max = math.ceil((cfg.LIDAR.X_RANGE[1]-cfg.LIDAR.X_RANGE[0])/cfg.LIDAR.VOXEL_LEN) - 1
        y_max = math.ceil((cfg.LIDAR.Y_RANGE[1]-cfg.LIDAR.Y_RANGE[0])/cfg.LIDAR.VOXEL_LEN) - 1
        z_max = math.ceil((cfg.LIDAR.Z_RANGE[1]-cfg.LIDAR.Z_RANGE[0])/cfg.LIDAR.VOXEL_HEIGHT) - 1
        area_3d = [[0,x_max],[0,y_max],[0,z_max]]
        anchor_3d_sizes = cfg.LIDAR.ANCHORS/([cfg.LIDAR.VOXEL_LEN,cfg.LIDAR.VOXEL_LEN,cfg.LIDAR.VOXEL_HEIGHT])
        anchor_stride = cfg.LIDAR.ANCHOR_STRIDE
        #ground_plane = cfg.LIDAR.GROUND_PLANE_COEFF

        return tile_anchors_3d(area_3d,
                               anchor_3d_sizes,
                               anchor_stride)


def tile_anchors_3d(area_extents,
                    anchor_3d_sizes,
                    anchor_stride):
    """
    Tiles anchors over the area extents by using meshgrids to
    generate combinations of (x, y, z), (l, w, h) and ry.

    Args:
        area_extents: [[min_x, max_x], [min_y, max_y], [min_z, max_z]]
        anchor_3d_sizes: list of 3d anchor sizes N x (l, w, h)
        anchor_stride: stride lengths (x_stride, z_stride)
        ground_plane: coefficients of the ground plane e.g. [0, -1, 0, 0]

    Returns:
        boxes: list of 3D anchors in box_3d format N x [x, y, z, l, w, h, ry]
    """
    # Convert sizes to ndarray
    anchor_3d_sizes = np.asarray(anchor_3d_sizes,dtype=(int))

    anchor_stride_x = anchor_stride[0]
    anchor_stride_y = anchor_stride[1]
    anchor_rotations = np.asarray([0, np.pi / 2.0])

    x_start = area_extents[0][0] + anchor_stride[0] / 2.0
    x_end = area_extents[0][1]
    x_centers = np.array(np.arange(x_start, x_end, step=anchor_stride_x),
                         dtype=np.float32)

    y_start = area_extents[2][1] - anchor_stride[1] / 2.0
    y_end = area_extents[2][0]
    y_centers = np.array(np.arange(y_start, y_end, step=-anchor_stride_y),
                         dtype=np.float32)

    # Use ranges for substitution
    size_indices = np.arange(0, len(anchor_3d_sizes))
    rotation_indices = np.arange(0, len(anchor_rotations))

    # Generate matrix for substitution
    # e.g. for two sizes and two rotations
    # [[x0, z0, 0, 0], [x0, z0, 0, 1], [x0, z0, 1, 0], [x0, z0, 1, 1],
    #  [x1, z0, 0, 0], [x1, z0, 0, 1], [x1, z0, 1, 0], [x1, z0, 1, 1], ...]
    before_sub = np.stack(np.meshgrid(x_centers,
                                      y_centers,
                                      size_indices,
                                      rotation_indices),
                          axis=4).reshape(-1, 4)

    # Place anchors on the ground plane
    #a, b, c, d = ground_plane
    all_x = before_sub[:, 0]
    all_y = before_sub[:, 1]
    all_z = np.zeros_like(all_x)
    #all_z = -(a * all_x + c * all_y + d) / b

    # Create empty matrix to return
    num_anchors = len(before_sub)
    all_anchor_boxes_3d = np.zeros((num_anchors, 7))

    # Fill in x, y, z
    all_anchor_boxes_3d[:, 0:3] = np.stack((all_x, all_y, all_z), axis=1)

    # Fill in shapes
    sizes = anchor_3d_sizes[np.asarray(before_sub[:, 2], np.int32)]
    all_anchor_boxes_3d[:, 3:6] = sizes

    # Fill in rotations
    rotations = anchor_rotations[np.asarray(before_sub[:, 3], np.int32)]
    all_anchor_boxes_3d[:, 6] = rotations

    return num_anchors, all_anchor_boxes_3d
