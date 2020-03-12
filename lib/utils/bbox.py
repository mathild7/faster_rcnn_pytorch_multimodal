import torch
import numpy as np


def bbox_overlaps(boxes, query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray or tensor or variable
    query_boxes: (K, 4) ndarray or tensor or variable
    Returns
    -------
    overlaps: (N, K) overlap between boxes and query_boxes
    """
    if isinstance(boxes, np.ndarray):
        boxes = torch.from_numpy(boxes)
        query_boxes = torch.from_numpy(query_boxes)
        out_fn = lambda x: x.numpy()  # If input is ndarray, turn the overlaps back to ndarray when return
    else:
        out_fn = lambda x: x

    box_areas = (boxes[:, 2] - boxes[:, 0] + 1) * \
            (boxes[:, 3] - boxes[:, 1] + 1)
    query_areas = (query_boxes[:, 2] - query_boxes[:, 0] + 1) * \
            (query_boxes[:, 3] - query_boxes[:, 1] + 1)

    iw = (torch.min(boxes[:, 2:3], query_boxes[:, 2:3].t()) - torch.max(
        boxes[:, 0:1], query_boxes[:, 0:1].t()) + 1).clamp(min=0)
    ih = (torch.min(boxes[:, 3:4], query_boxes[:, 3:4].t()) - torch.max(
        boxes[:, 1:2], query_boxes[:, 1:2].t()) + 1).clamp(min=0)
    ua = box_areas.view(-1, 1) + query_areas.view(1, -1) - iw * ih
    overlaps = iw * ih / ua
    return out_fn(overlaps)

#Input: X1,Y1,X2,Y2
#Output: [[x1,y1],[x1,y2],[x2,y2][x2,y1]]
def bbox_bev_2pt_to_4_pt(x1,y1,x2,y2):
    p = []
    #p0 -> bottom, left
    p.append([x1,y1])
    #p1 -> bottom, right
    p.append([x1,y2])
    #p2 -> top, right
    p.append([x2,y2])
    #p3 -> top, left
    p.append([x2,y1])
    return p

#Input: [bottomleft,bottomright,topright,topleft]
def rotated_4p_to_axis_aligned_2p(points):
    bottomleft  = points[0]
    bottomright = points[1]
    topright    = points[2]
    topleft     = points[3]
    x1 = min(bottomleft[0],topleft[0])
    x2 = max(bottomright[0],topright[0])
    y1 = min(bottomleft[1],bottomright[1])
    y2 = max(topleft[1],topright[1])
    return [x1,y1,x2,y2]

#Input is (xc,yc,zc,l,w,h,ry)
#This creates a BEV box that contains an entire rotated object (Birdnet Fig 2)
def bbox_3d_to_bev_axis_aligned(bboxes):
    bev_bboxes = []
    for bbox in bboxes:
        x1 = bbox[0] - bbox[3]/2
        x2 = bbox[0] + bbox[3]/2
        y1 = bbox[1] - bbox[4]/2
        y2 = bbox[2] + bbox[4]/2
        points = bbox_bev_2pt_to_4_pt(x1, y1, x2, y2)
        p_c    = [bbox[0], bbox[1]]
        rot_points = rotate_in_bev(points, p_c, bbox[6])
        axis_aligned_points = rotated_4p_to_axis_aligned_2p(rot_points)
        bev_bboxes.append(axis_aligned_points)
    return np.asarray(bev_bboxes)
#(xc,yc,zc,l,w,h,ry)
def bbox_to_voxel_grid(bboxes,bev_extants,info):
    bboxes[:,0] = np.clip((bboxes[:,0]-bev_extants[0])*(bev_extants[3]-bev_extants[0])/(info[1]-info[0]),info[0],info[1])
    bboxes[:,1] = np.clip((bboxes[:,0]-bev_extants[1])*(bev_extants[4]-bev_extants[1])/(info[1]-info[0]),info[2],info[3])
    bboxes[:,2] = np.clip((bboxes[:,1]-bev_extants[2])*(bev_extants[5]-bev_extants[2])/(info[3]-info[2]),info[4],info[5])
    bboxes[:,3] = np.clip((bboxes[:,3])*(bev_extants[3]-bev_extants[0])/(info[1]-info[0]),info[0],info[1])
    bboxes[:,4] = np.clip((bboxes[:,3])*(bev_extants[4]-bev_extants[1])/(info[1]-info[0]),info[0],info[1])
    bboxes[:,5] = np.clip((bboxes[:,3])*(bev_extants[5]-bev_extants[2])/(info[3]-info[2]),info[2],info[3])
    return bboxes

#Input is (xc,yc,zc,l,w,h,ry)
def bbox_3d_to_bev_4pt(bboxes):
    bev_bboxes = []
    for bbox in bboxes:
        x1 = bbox[0] - bbox[3]/2
        x2 = bbox[0] + bbox[3]/2
        y1 = bbox[1] - bbox[4]/2
        y2 = bbox[2] + bbox[4]/2
        points = bbox_bev_2pt_to_4_pt(x1, y1, x2, y2)
        p_c    = [bbox[0], bbox[1]]
        rot_points = rotate_in_bev(points, p_c, bbox[6])
        bev_bboxes.append(rot_points)
    return np.asarray(bev_bboxes)

def bbox_overlaps_3d(boxes, query_boxes):
    overlaps = np.zeros((boxes.shape[0],query_boxes.shape[0]))
    for i, box in enumerate(boxes):
        overlaps[i] = three_d_iou(box,query_boxes)
    return overlaps

#STOLEN FROM WAVEDATA KUJASON https://github.com/kujason/wavedata :-)
def three_d_iou(box, boxes):
    """Computes approximate 3D IOU between a 3D bounding box 'box' and a list
    of 3D bounding boxes 'boxes'. All boxes are assumed to be aligned with
    respect to gravity. Boxes are allowed to rotate only around their z-axis.

    :param box: a numpy array of the form: [ry, l, h, w, tx, ty, tz]
    :param boxes: a numpy array of the form:
        [[ry, l, h, w, tx, ty, tz], [ry, l, h, w, tx, ty, tz]]

    :return iou: a numpy array containing 3D IOUs between box and every element
        in numpy array boxes.
    """
    # First, rule out boxes that do not intersect by checking if the spheres
    # which inscribes them intersect.

    if len(boxes.shape) == 1:
        boxes = np.array([boxes])

    box_diag = np.sqrt(np.square(box[1]) +
                       np.square(box[2]) +
                       np.square(box[3])) / 2

    boxes_diag = np.sqrt(np.square(boxes[:, 1]) +
                         np.square(boxes[:, 2]) +
                         np.square(boxes[:, 3])) / 2

    dist = np.sqrt(np.square(boxes[:, 4] - box[4]) +
                   np.square(boxes[:, 5] - box[5]) +
                   np.square(boxes[:, 6] - box[6]))

    non_empty = box_diag + boxes_diag >= dist

    iou = np.zeros(len(boxes), np.float64)

    if non_empty.any():
        height_int, _ = height_metrics(box, boxes[non_empty])
        rect_int = get_rectangular_metrics(box, boxes[non_empty])

        intersection = np.multiply(height_int, rect_int)

        vol_box = np.prod(box[1:4])

        vol_boxes = np.prod(boxes[non_empty, 1:4], axis=1)

        union = vol_box + vol_boxes - intersection

        iou[non_empty] = intersection / union

    if iou.shape[0] == 1:
        iou = iou[0]

    return iou

#STOLEN FROM AVOD :-)
def rotate_in_bev(p, p_c, rot):
    points = np.asarray(p,dtype=np.float32)
    center = np.asarray(p_c)
    pts    = np.asarray(p)
    rot_mat = np.reshape([[np.cos(rot), np.sin(rot)],
                        [-np.sin(rot), np.cos(rot)]],
                        (2, 2))
    box_p = []
    for coords in pts:
        rotated = np.dot(rot_mat,coords-center) + center
        box_p.append(rotated)
    #rot_points = np.dot(rot_mat,points) + box_xy
    #box_p0 = (np.dot(rot_mat, p0) + box_xy)
    #box_p1 = (np.dot(rot_mat, p1) + box_xy)
    #box_p2 = (np.dot(rot_mat, p2) + box_xy)
    #box_p3 = (np.dot(rot_mat, p3) + box_xy)

    #box_points = np.array([box_p0, box_p1, box_p2, box_p3])

    # Calculate normalized box corners for ROI pooling
    #x_extents_min = bev_extents[0][0]
    #y_extents_min = bev_extents[1][1]  # z axis is reversed
    #points_shifted = box_points - [x_extents_min, y_extents_min]

    #x_extents_range = bev_extents[0][1] - bev_extents[0][0]
    #y_extents_range = bev_extents[1][0] - bev_extents[1][1]
    #box_points_norm = points_shifted / [x_extents_range, y_extents_range]

    box_points = np.asarray(box_p, dtype=np.float32)
    #box_points_norm = np.asarray(box_points_norm, dtype=np.float32)

    return box_points
