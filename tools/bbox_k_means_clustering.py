#https://lars76.github.io/object-detection/k-means-anchor-boxes/
import json
import os
import numpy as np
from enum import Enum
class class_enum(Enum):
    UNKNOWN = 0
    VEHICLE = 1
    PEDESTRIAN = 2
    SIGN = 3
    CYCLIST = 4
def iou(box, clusters):
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]
    #print(box_area)
    #print(cluster_area)
    #print(intersection)
    iou_ = intersection / (box_area + cluster_area - intersection)

    return iou_

def kmeans(boxes, k, dist=np.median):
    rows = boxes.shape[0]
    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    np.random.seed()

    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)

        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break

        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters


if __name__ == '__main__':
    path = '/home/mat/thesis/data/waymo'
    mode = 'train'
    labels_filename = os.path.join(path, mode,'labels/labels.json')
    ix = 0
    w = 1920.0
    h = 730.0
    with open(labels_filename,'r') as labels_file:
        data = labels_file.read()
        #print(data)
        #print(data)
        labels = json.loads(data)
        for img_labels in labels:
            for i, bbox in enumerate(img_labels['box']):
                ix += 1
        gt_roidb = np.zeros((ix,2))
        ix = 0
        #print('num labels {}'.format(len(labels)))
        for img_labels in labels:
            for i, bbox in enumerate(img_labels['box']):
                anno_cat   = img_labels['class'][i]
                if(class_enum(anno_cat) == class_enum.SIGN):
                    anno_cat = class_enum.UNKNOWN.value
                elif(class_enum(anno_cat) == class_enum.CYCLIST):
                    #Sign is taking index 3, where my code expects cyclist to be. Therefore replace any cyclist (class index 4) with sign (class index 3)
                    anno_cat = class_enum.SIGN.value

                #OVERRIDE
                if(class_enum(anno_cat) != class_enum.VEHICLE):
                    anno_cat = class_enum.UNKNOWN.value
                    continue
                x1 = float(bbox['x1'])
                y1 = float(bbox['y1'])
                x2 = float(bbox['x2'])
                y2 = float(bbox['y2'])
                if(x2 - x1 <= 10):
                    continue
                elif(y2 - y1 <= 10):
                    continue
                bbox_n = [(x2-x1)/w,(y2-y1)/h]
                if(bbox_n[0] < 0.001 and bbox_n[1] < 0.001):
                    continue
                ix += 1
                gt_roidb[ix] = bbox_n
    gt_roidb = gt_roidb[0:ix,:]
    clusters = kmeans(gt_roidb,4)
    clusters = clusters*[w,h]
    print(clusters)
