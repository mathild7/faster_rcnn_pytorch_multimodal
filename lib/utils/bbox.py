import torch
import numpy as np

#TODO: Add min height 
def bbox_overlaps(boxes, query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray or tensor or variable (ROI/Prediction)
    query_boxes: (K, 4) ndarray or tensor or variable (Ground Truth)
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
Function: bbox_3d_to_bev_axis_aligned
Computes the axis-aligned tightest bounding box required to fit a rotated (along ry) bounding box in BEV
Arguments:
----------
bbox -> (Nx7) [XC,YC,ZC,L,W,H,RY]
width -> BEV image width
height -> BEV image height
Returns:
---------
clipped_bev_bboxes -> (Nx4) [X1,Y1,X2,Y2]
"""
def bbox_3d_to_bev_axis_aligned(bbox,width=0,height=0):
    points = bbox_3d_to_bev_4pt(bbox)
    x1 = np.min(points[:,:,0],axis=1)
    x2 = np.max(points[:,:,0],axis=1)
    y1 = np.min(points[:,:,1],axis=1)
    y2 = np.max(points[:,:,1],axis=1)
    bev_bboxes = np.swapaxes(np.asarray([x1,y1,x2,y2],dtype=np.float32),1,0)
    #TODO: Ignore bboxes that have been clipped to be too small? Probably need to do this at an earlier stage
    clipped_bev_bboxes = _bbox_clip(width,height,bev_bboxes)
    return clipped_bev_bboxes

"""
function: bbox_3t_to_8pt
input: (N,7) 3d bounding boxes
output: (N,8,3) 3d bboxes in 8pt format
format: (clockwise) start at back,left,top end at front,left,bottom

"""
def bbox_3d_to_8pt(bbox):
    #bbox center
    x_c = bbox[:, 0]
    y_c = bbox[:, 1]
    z_c = bbox[:, 2]
    #bbox size
    l   = bbox[:, 3]
    w   = bbox[:, 4]
    h   = bbox[:, 5]
    #bbox heading
    rot = bbox[:, 6]
    x_corners = np.asarray([ l/2,l/2,-l/2,-l/2, l/2, l/2,-l/2,-l/2])[:,:,np.newaxis]
    y_corners = np.asarray([-w/2,w/2, w/2,-w/2,-w/2, w/2, w/2,-w/2])[:,:,np.newaxis]
    z_corners = np.asarray([ h/2,h/2, h/2, h/2,-h/2,-h/2,-h/2,-h/2])[:,:,np.newaxis]
    corners_3d = np.stack((x_corners,y_corners,z_corners),axis=2)
    p = corners_3d
    #centers_3d = np.stack((x_c,y_c,z_c),axis=0)
    p[:,:,0] = p[:,:,0]*np.cos(rot) - p[:,:,1]*np.sin(rot)
    p[:,:,1] = p[:,:,0]*np.sin(rot) + p[:,:,1]*np.cos(rot)
    p[:,:,2] = p[:,:,2]
    #corners_3d = rotate_in_3d(corners_3d,rot)
    corners_3d = p
    corners_3d[0,:] = corners_3d[0,:] + x_c
    corners_3d[1,:] = corners_3d[1,:] + y_c
    corners_3d[2,:] = corners_3d[2,:] + z_c
    return corners_3d

#Clip bounding boxes to pixel array
def _bbox_clip(width,height,bboxes):
    bboxes[:,0] = np.clip(bboxes[:,0],0,width)
    bboxes[:,2] = np.clip(bboxes[:,2],0,width)
    bboxes[:,1] = np.clip(bboxes[:,1],0,height)
    bboxes[:,3] = np.clip(bboxes[:,3],0,height)
    return bboxes


"""
Function: bbox_pc_to_voxel_grid
shifts and scales bounding boxes to fit on the voxel grid image interpretation of the LiDAR scan
Arguments:
----------
bboxes      -> (Nx7) [XC,YC,ZC,L,W,H,RY]
bev_extants -> LiDAR scan range [x1,y1,z1,x2,y2,z2]
info        -> BEV voxel grid image size (x_min,x_max,y_min,y_max,z_min,z_max)
Returns:
---------
vg_bboxes -> (Nx7) [XC(mod),YC(mod),ZC,L(mod),W(mod),H,ry] (scaled and shifted to fit the image)
"""
def bbox_pc_to_voxel_grid(bboxes,bev_extants,info):
    #vg_bboxes = np.zeros((bboxes.shape[0],4))
    #Make it scale invariant
    scale = info[6]
    #Ensure that the transform is done assuming bboxes are in their full scale
    s_info = np.asarray(info[0:6]) * 1/scale
    bboxes[:,0] = (bboxes[:,0]-bev_extants[0])*((s_info[1]-s_info[0])/(bev_extants[3]-bev_extants[0]))
    bboxes[:,1] = (bboxes[:,1]-bev_extants[1])*((s_info[3]-s_info[2])/(bev_extants[4]-bev_extants[1]))
    bboxes[:,3] = (bboxes[:,3])*((s_info[1]-s_info[0])/(bev_extants[3]-bev_extants[0]))
    bboxes[:,4] = (bboxes[:,4])*((s_info[3]-s_info[2])/(bev_extants[4]-bev_extants[1]))
    #DEPRECATED
    #bboxes[:,2] = bboxes[:,2] - bev_extants[2]
    return bboxes


"""
Function: bbox_voxel_grid_to_pc
shifts and scales bounding boxes to fit on the LiDAR scan boundaries
Arguments:
----------
vg_bboxes   -> (Nx7) [XC,YC,ZC,L,W,H,RY]
bev_extants -> LiDAR scan range [x1,y1,z1,x2,y2,z2]
info        -> BEV voxel grid image size (x_min,x_max,y_min,y_max,z_min,z_max)
Returns:
---------
pc_bboxes -> (Nx7) [XC(mod),YC(mod),ZC,L(mod),W(mod),H,ry] (scaled and shifted to fit lidar extants)
"""
def bbox_voxel_grid_to_pc(bboxes,bev_extants,info,aabb=False):
    #vg_bboxes = np.zeros((bboxes.shape[0],4))
    scale = info[6]
    #Ensure that transform is done assuming bboxes are in their full scale
    s_info = np.asarray(info[0:6]) * 1/scale
    if(aabb):
        #X1
        bboxes[:,0] = (bboxes[:,0])*((bev_extants[3]-bev_extants[0])/(s_info[1]-s_info[0])) + bev_extants[0]
        #Y1
        bboxes[:,1] = (bboxes[:,1])*((bev_extants[4]-bev_extants[1])/(s_info[3]-s_info[2])) + bev_extants[1]
        #X2
        bboxes[:,2] = (bboxes[:,2])*((bev_extants[3]-bev_extants[0])/(s_info[1]-s_info[0])) + bev_extants[0]
        #Y2
        bboxes[:,3] = (bboxes[:,3])*((bev_extants[4]-bev_extants[1])/(s_info[3]-s_info[2])) + bev_extants[1]
    #X,Y,Z,L,W,H
    else:
        bboxes[:,0] = (bboxes[:,0])*((bev_extants[3]-bev_extants[0])/(s_info[1]-s_info[0])) + bev_extants[0]
        bboxes[:,1] = (bboxes[:,1])*((bev_extants[4]-bev_extants[1])/(s_info[3]-s_info[2])) + bev_extants[1]
        bboxes[:,3] = (bboxes[:,3])*((bev_extants[3]-bev_extants[0])/(s_info[1]-s_info[0]))
        bboxes[:,4] = (bboxes[:,4])*((bev_extants[4]-bev_extants[1])/(s_info[3]-s_info[2]))
        #DEPRECATED - height starts at 0
        #bboxes[:,2] = bboxes[:,2] + bev_extants[2]
    return bboxes

"""
Function: bbox_3d_to_bev_4pt
transforms a 3d bounding box to a 4pt interpretation of the BEV bbox
Arguments:
----------
bboxes      -> (Nx7) [XC,YC,ZC,L,W,H,RY]
Returns:
---------
bev_bboxes -> (Nx4x2) [[X1, Y1], [X2,Y2], [X3, Y3], [X4, Y4]]
"""
def bbox_3d_to_bev_4pt(bboxes):
    x1 = bboxes[:,0] - bboxes[:,3]/2
    x2 = bboxes[:,0] + bboxes[:,3]/2
    y1 = bboxes[:,1] - bboxes[:,4]/2
    y2 = bboxes[:,1] + bboxes[:,4]/2
    points = _bbox_bev_2pt_to_4pt(x1, y1, x2, y2)
    p_c    = np.hstack((bboxes[:,0:1], bboxes[:,1:2]))
    bev_bboxes = rotate_in_bev(points, p_c, bboxes[:,6])
    return np.asarray(bev_bboxes)

#Helper function for above
#Input: X1,Y1,X2,Y2
#Output: [[x1,y1],[x1,y2],[x2,y2][x2,y1]]
def _bbox_bev_2pt_to_4pt(x1,y1,x2,y2):
    p = []
    #p0 -> bottom, left
    p.append([x1,y1])
    #p1 -> bottom, right
    p.append([x1,y2])
    #p2 -> top, right
    p.append([x2,y2])
    #p3 -> top, left
    p.append([x2,y1])
    return np.transpose(np.asarray(p),(2,0,1))

def bbox_overlaps_3d(boxes, query_boxes):
    overlaps = np.zeros((boxes.shape[0],query_boxes.shape[0]))
    for i, box in enumerate(boxes):
        overlaps[i] = three_d_iou(box,query_boxes)
    return overlaps

"""
Function: rotate_in_bev
For taking in a BEV bounding box and rotating along yaw
Arguments:
--------
p -> 4 pt encoding of axis aligned bboxes (Nx4x2)
p_c -> list of center points for each bbox (Nx2)
rot -> list of updated headings for each bbox (N)
Return:
--------
box_points -> 4pt encoding of rotated bboxes(Nx4x2)
"""
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

def rotate_in_3d(p, p_c, rot):
    rot = rot[:,np.newaxis].repeat(p.shape[1],axis=1)
    p[:,:,0] = p[:,:,0]*np.cos(rot) - p[:,:,1]*np.sin(rot)
    p[:,:,1] = p[:,:,0]*np.sin(rot) + p[:,:,1]*np.cos(rot)
    p[:,:,2] = p[:,:,2]
    return p


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
def bbaa_graphics_gems(bboxes,width,height,clip=True):

    rot = bboxes[:,6:7]
    M = np.asarray([[np.cos(rot), np.sin(rot)],[-np.sin(rot), np.cos(rot)]]).squeeze(-1).transpose((2,0,1))
    T = bboxes[:,0:2]
    A = bboxes
    Amin = np.zeros((A.shape[0],2))
    Amax = np.zeros((A.shape[0],2))
    Bmin = np.zeros((A.shape[0],2))
    Bmax = np.zeros((A.shape[0],2))
    #Copy box A into a min array and a max array for easy reference.

    Amin[:,0] = - A[:,3]/2.0
    Amax[:,0] = + A[:,3]/2.0
    Amin[:,1] = - A[:,4]/2.0
    Amax[:,1] = + A[:,4]/2.0

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
    if(clip):
        B = _bbox_clip(width-1,height-1,bev_bboxes)
    else:
        B = bev_bboxes
    return B


def bbaa_graphics_gems_torch(bboxes,width,height,clip=True):

    rot = bboxes[:,6] #.cpu().numpy()
    rot_top = torch.cat((torch.cos(rot).unsqueeze(0),torch.sin(rot).unsqueeze(0)),dim=0)
    rot_bot = torch.cat((-torch.sin(rot).unsqueeze(0),torch.cos(rot).unsqueeze(0)),dim=0)
    rot = torch.stack((rot_top,rot_bot),dim=0)
    M = rot.permute((2,0,1))
    T = bboxes[:,0:2]
    A = bboxes
    Amin = torch.zeros((A.shape[0],2)).to(device=bboxes.device)
    Amax = torch.zeros((A.shape[0],2)).to(device=bboxes.device)
    Bmin = torch.zeros((A.shape[0],2)).to(device=bboxes.device)
    Bmax = torch.zeros((A.shape[0],2)).to(device=bboxes.device)
    #Copy box A into a min array and a max array for easy reference.

    Amin[:,0] = - A[:,3]/2.0
    Amax[:,0] = A[:,3]/2.0
    Amin[:,1] = - A[:,4]/2.0
    Amax[:,1] = A[:,4]/2.0

    #Now find the extreme points by considering the product of the
    #min and max with each component of M.
    a = torch.einsum('ijk,ik->ijk',M,Amin)
    b = torch.einsum('ijk,ik->ijk',M,Amax)
    Bmin = torch.min(a,b)
    Bmax = torch.max(a,b)
    Bmin = torch.sum(Bmin,dim=2).type(torch.float32)
    Bmax = torch.sum(Bmax,dim=2).type(torch.float32)
    #Copy the result into the new box.
    Bmin = Bmin + T
    Bmax = Bmax + T
    x1 = Bmin[:,0:1]
    x2 = Bmax[:,0:1]
    y1 = Bmin[:,1:2]
    y2 = Bmax[:,1:2]
    bev_bboxes = torch.cat((x1,y1,x2,y2),dim=1)
    if(clip):
        B = _bbox_clip(width-1,height-1,bev_bboxes)
    else:
        B = bev_bboxes
    return B


def draw_bev_bbox(draw,bbox,bev_shape,transform=True,colors=None):
    bboxes = bbox[np.newaxis,:]
    if(colors is None):
        colors = [255,255,255]
    colors  = np.asarray(colors)[np.newaxis,:]
    draw_bev_bboxes(draw,bboxes,bev_shape,transform,colors)

def draw_bev_bboxes(draw,bboxes,bev_shape,transform=True,colors=None):
    bboxes_4pt = bbox_3d_to_bev_4pt(bboxes)
    #TODO: if clip, keep same angle
    bboxes_4pt[:,:,0] = np.clip(bboxes_4pt[:,:,0],0,bev_shape[0]-1)
    bboxes_4pt[:,:,1] = np.clip(bboxes_4pt[:,:,1],0,bev_shape[1]-1)
    bboxes_4pt = bboxes_4pt.astype(dtype=np.int64)
    z1 = bboxes[:,2]-bboxes[:,5]
    z2 = bboxes[:,2]+bboxes[:,5]
    z_max = bev_shape[2] - 1
    if(colors is None):
        c = np.clip(float(z2/float(z_max))*255,0,255).astype(dtype='uint8')
        c = [c,c,c]
    else:
        c = colors

    for i, bbox in enumerate(bboxes_4pt):
        draw_polygon(draw,bbox,c[i])

def draw_polygon(draw,pixel_coords,c):
    for i in range(len(pixel_coords)):
        if(i == 0):
            xy1 = pixel_coords[i]
            xy2 = pixel_coords[len(pixel_coords)-1]
        else:
            xy1 = pixel_coords[i]
            xy2 = pixel_coords[i-1]
        #if(xy1[0] >= self._draw_width or xy1[0] < 0):
        #    print(xy1)
        #if(xy2[0] >= self._draw_width or xy2[0] < 0):
        #    print(xy2)
        #print('drawing: {}-{}'.format(xy1,xy2))
        #line = np.concatenate((xy1,xy2))
        draw.line([xy1[0],xy1[1],xy2[0],xy2[1]],fill=(c[0],c[1],c[2]),width=2)
        draw.point(xy1,fill=(c[0],c[1],c[2]))