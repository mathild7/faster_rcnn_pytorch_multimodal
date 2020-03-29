import torch
import numpy as np
import os
from PIL import Image, ImageDraw
# XC, YC, ZC, L, W, H, Ry
def bbox_rot_debug():
    bboxes = [[350,400,2,50,20,3, np.pi/8],[400,300,2,50,20,3, np.pi/8],[100,100,2,50,20,3, np.pi/8]]
    bboxes = np.asarray(bboxes)
    datapath = '/home/mat/thesis/data/waymo/debug/'
    out_file = os.path.join(datapath,'test_target.png')
    #lidb = waymo_lidb()
    #Extract voxel grid size
    width   = 700
    #Y is along height axis in image domain
    height  = 800
    blank_canvas = np.zeros((height,width,3),dtype=np.uint8)
    img = Image.fromarray(blank_canvas)
    draw = ImageDraw.Draw(img)
    for bbox in bboxes:
        x1 = bbox[0]-bbox[3]/2
        x2 = bbox[0]+bbox[3]/2
        y1 = bbox[1]-bbox[4]/2
        y2 = bbox[1]+bbox[4]/2
        draw.rectangle((x1,y1,x2,y2),outline=(0,255,0))
    #bboxes_4pt = bbox_3d_to_bev_4pt(bboxes)
    #bbox_4pt = bboxes_4pt[0]
    #BL
    #tl = bbox_4pt[0]
    #bl = bbox_4pt[1]
    #br = bbox_4pt[2]
    #tr = bbox_4pt[3]
    #l1 = [(tl[0],tl[1]),(bl[0],bl[1])]
    #l1 = [tl[0],tl[1],bl[0],bl[1]]
    #l2 = [bl[0],bl[1],br[0],br[1]]
    #l3 = [br[0],br[1],tr[0],tr[1]]
    #l4 = [tr[0],tr[1],tl[0],tl[1]]
    #draw.line(l1,fill=(255,0,0))
    #draw.line(l2,fill=(255,0,0))
    #draw.line(l3,fill=(255,0,0))
    #draw.line(l4,fill=(255,0,0))
    #aa_bboxes = bbox_3d_to_bev_axis_aligned(bboxes,width,height)
    aa_bboxes = bbaa_graphics_gems(bboxes,width,height).astype(dtype=np.uint64)
    for aa_bbox in aa_bboxes:
        print(aa_bbox)
        draw.rectangle([(aa_bbox[0],aa_bbox[1]),(aa_bbox[2],aa_bbox[3])],outline=(0,0,255))
    print('Saving BEV map file at location {}'.format(out_file))
    img.save(out_file,'png')

"""  
Transforming Axis-Aligned Bounding Boxes
by Jim Arvo
from "Graphics Gems", Academic Press, 1990


Transforms a 2D axis-aligned box via a 2x2 matrix and a translation
vector and returns an axis-aligned box enclosing the result.
Matrix3  M;  	/* Transform matrix.             */
Vector3  T;  	/* Translation matrix.           */
Box3     A;  	/* The original bounding box.    */
Box3    *B;  	/* The transformed bounding box. */
"""
def bbaa_graphics_gems(bboxes,width,height):

    rot = bboxes[:,6:7]
    M = np.asarray([[np.cos(rot), np.sin(rot)],[-np.sin(rot), np.cos(rot)]]).squeeze(-1).transpose((2,0,1))
    T = bboxes[:,0:2]
    A = bboxes
    Amin = np.zeros((A.shape[0],2))
    Amax = np.zeros((A.shape[0],2))
    Bmin = np.zeros((A.shape[0],2))
    Bmax = np.zeros((A.shape[0],2))
    #Copy box A into a min array and a max array for easy reference.

    Amin = - A[:,3:5]/2.0
    Amax = + A[:,3:5]/2.0

    #Now find the extreme points by considering the product of the
    #min and max with each component of M.
    a = np.einsum('ijk,ik->ijk',M,Amin)
    b = np.einsum('ijk,ik->ijk',M,Amax)
    Bmin = np.minimum(a,b)
    Bmax = np.maximum(a,b)
    Bmin = np.sum(Bmin,axis=2).astype(dtype=np.float32)
    Bmax = np.sum(Bmax,axis=2).astype(dtype=np.float32)
    #Copy the result into the new box.
    Bmin = Bmin + T
    Bmax = Bmax + T
    x1 = Bmin[:,0:1]
    x2 = Bmax[:,0:1]
    y1 = Bmin[:,1:2]
    y2 = Bmax[:,1:2]
    bev_bboxes = np.concatenate((x1,y1,x2,y2),axis=1)
    B = bbox_clip(width-1,height-1,bev_bboxes)
    return B






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
    test = boxes[:, 2:3]
    iw = (torch.min(boxes[:, 2:3], query_boxes[:, 2:3].t()) - torch.max(
        boxes[:, 0:1], query_boxes[:, 0:1].t()) + 1).clamp(min=0)
    ih = (torch.min(boxes[:, 3:4], query_boxes[:, 3:4].t()) - torch.max(
        boxes[:, 1:2], query_boxes[:, 1:2].t()) + 1).clamp(min=0)
    ua = box_areas.view(-1, 1) + query_areas.view(1, -1) - iw * ih
    overlaps = iw * ih / ua
    return out_fn(overlaps)
"""
bbox bev 2pt to 4pt
Parameters:
-------
x1,y1,x2,y2: Each being a list (x1,y1) is top left

Returns:
-------
4 point interpretation of bounding box
[Top Left],[Bottom Left],[Bottom Right], [Top Right]
"""
def bbox_bev_2pt_to_4pt(x1,y1,x2,y2):
    p = []
    p.append([x1,y1])
    p.append([x1,y2])
    p.append([x2,y2])
    p.append([x2,y1])
    return np.transpose(np.asarray(p),(2,0,1))

#Input: [bottomleft,bottomright,topright,topleft]
def rotated_4p_to_axis_aligned_2p(points):
    topright    = points[0]
    topleft     = points[1]
    bottomleft  = points[2]
    bottomright = points[3]
    points = np.asarray(points)
    x1 = min(points[:,0])
    x2 = max(points[:,0])
    y1 = min(points[:,1])
    y2 = max(points[:,1])
    return [x1,y1,x2,y2]

#Input is (xc,yc,zc,l,w,h,ry)
#This creates a BEV box that contains an entire rotated object (Birdnet Fig 2)
def bbox_3d_to_bev_axis_aligned(bbox,width=0,height=0):
    #bev_bboxes = []
    #for bbox in bboxes:
    #x1 = bbox[:,0] - bbox[:,3]/2
    #x2 = bbox[:,0] + bbox[:,3]/2
    #y1 = bbox[:,1] - bbox[:,4]/2
    #y2 = bbox[:,1] + bbox[:,4]/2
    xl = bbox[:,3]
    yw = bbox[:,4]
    ry = np.abs(bbox[:,6])
    x_normal_rot = xl*np.cos(ry) - yw*np.sin(ry)
    x_swapped_rot = yw*np.cos(ry - np.pi/2.0)  - xl*np.sin(ry - np.pi/2.0)
    rot_lx = np.abs(np.where(ry <= np.pi/2.0, x_normal_rot, x_swapped_rot))
    y_normal_rot = xl*np.sin(ry) + yw*np.cos(ry)
    y_swapped_rot = yw*np.sin(ry - np.pi/2.0) + xl*np.cos(ry - np.pi/2.0)
    rot_yw = np.abs(np.where(ry <= np.pi/2.0, y_normal_rot, y_swapped_rot))
    
    #points = bbox_3d_to_bev_4pt(bbox)
    #x1 = np.min(points[:,:,0],axis=1)
    #x2 = np.max(points[:,:,0],axis=1)
    #y1 = np.min(points[:,:,1],axis=1)
    #y2 = np.max(points[:,:,1],axis=1)
    #h_dir = np.where(bbox[:,6] > 0, 1, -1)
    #x_diff = np.abs(x2-x1)
    #y_diff = np.abs(y2-y1)
    #lx  = x_diff*np.cos(ry) - y_diff*np.sin(ry)
    #wy  = x_diff*np.sin(ry) + y_diff*np.cos(ry)
    #points = bbox_bev_2pt_to_4pt(x1, y1, x2, y2)
    #p_c    = [bbox[0], bbox[1]]
    #WARNING: currently skipping rotation
    #rot_points = rotate_in_bev(points, p_c, bbox[6])
    #axis_aligned_points = rotated_4p_to_axis_aligned_2p(rot_points)
    #X1,Y1,X2,Y2
    x1 = bbox[:,0] - rot_lx/2
    x2 = bbox[:,0] + rot_lx/2
    y1 = bbox[:,1] - rot_yw/2
    y2 = bbox[:,1] + rot_yw/2
    bev_bboxes = np.swapaxes(np.asarray([x1,y1,x2,y2],dtype=np.float32),1,0)
    #TODO: Ignore bboxes that have been clipped to be too small?
    clipped_bev_bboxes = bbox_clip(width,height,bev_bboxes)
    #axis_aligned_points = [x1,y1,x2,y2]
    #bev_bboxes.append(axis_aligned_points)
    return clipped_bev_bboxes

def bbox_clip(width,height,bboxes):
    bboxes[:,0] = np.clip(bboxes[:,0],0,width)
    bboxes[:,2] = np.clip(bboxes[:,2],0,width)
    bboxes[:,1] = np.clip(bboxes[:,1],0,height)
    bboxes[:,3] = np.clip(bboxes[:,3],0,height)
    return bboxes

#(xc,yc,zc,l,w,h,ry)
#extants [x1,y1,z1,x2,y2,z2]
#info: voxel grid size (x_min,x_max,y_min,y_max,z_min,z_max)
def bbox_to_voxel_grid(bboxes,bev_extants,info):
    bboxes[:,0] = (bboxes[:,0]-bev_extants[0])*((info[1]-info[0]+1)/(bev_extants[3]-bev_extants[0]))
    #Invert Y as left is +40 but left is actually interpreted as most negative value in PC when converted to voxel grid.
    #bboxes[:,1] = -bboxes[:,1]
    bboxes[:,1] = (bboxes[:,1]-bev_extants[1])*((info[3]-info[2]+1)/(bev_extants[4]-bev_extants[1]))
    #bboxes[:,2] = (bboxes[:,2]-bev_extants[2])*((info[4]-info[5]+1)/(bev_extants[5]-bev_extants[2]))
    bboxes[:,3] = (bboxes[:,3])*((info[1]-info[0]+1)/(bev_extants[3]-bev_extants[0]))
    bboxes[:,4] = (bboxes[:,4])*((info[3]-info[2]+1)/(bev_extants[4]-bev_extants[1]))
    #bboxes[:,5] = (bboxes[:,5])*((info[4]-info[5]+1)/(bev_extants[5]-bev_extants[2]))
    #If inverting y axis, also must invert ry
    #bboxes[:,6] = -bboxes[:,6]
    return bboxes

#Input is (xc,yc,zc,l,w,h,ry)
def bbox_3d_to_bev_4pt(bbox):
    x1 = bbox[:,0] - bbox[:,3]/2
    x2 = bbox[:,0] + bbox[:,3]/2
    y1 = bbox[:,1] - bbox[:,4]/2
    y2 = bbox[:,1] + bbox[:,4]/2
    points = bbox_bev_2pt_to_4pt(x1, y1, x2, y2)
    p_c    = np.hstack((bbox[:,0:1], bbox[:,1:2]))
    bev_bboxes = rotate_in_bev(points, p_c, bbox[:,6])
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
    p_c = p_c[:,np.newaxis,:].repeat(p.shape[1],axis=1)
    rot = rot[:,np.newaxis].repeat(p.shape[1],axis=1)
    delta_x = p[:,:,0] - p_c[:,:,0]
    delta_y = p[:,:,1] - p_c[:,:,1]
    rot_x = delta_x*np.cos(rot) - delta_y*np.sin(rot)
    rot_y = delta_x*np.sin(rot) + delta_y*np.cos(rot)
    p[:,:,0] = rot_x + p_c[:,:,0]
    p[:,:,1] = rot_y + p_c[:,:,1]
    points = np.asarray(p,dtype=np.float32)
    center = np.asarray(p_c)
    pts    = np.asarray(p)
    #rot_mat = np.reshape([[np.cos(rot), np.sin(rot)],
    #                    [-np.sin(rot), np.cos(rot)]],
    #                    (2, 2))
    box_points = np.asarray(p, dtype=np.float32)
    return box_points

if __name__ == '__main__':
    bbox_rot_debug()