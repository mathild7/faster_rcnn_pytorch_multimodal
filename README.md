Faster-RCNN Image/LiDAR nets by Mat Hildebrand
Example run commands:
-----------------------------------------------------------------------------------
Lidar
First run
python3 ./trainval_net.py --net_type lidar --train_iter 1 --preload 0 --en_full_net 0

then after some training cycles run: (en_full_net is 1 by default)
python3 ./trainval_net.py --net_type lidar --weights_file ~/thesis/data/weights/lidar_rpn_60k.pth --train_iter 2 --preload 2

Current L2 mAP for vehicles: 0.64 (BEV) 0.25 (3D)
------------------------
Current sample commands:

RPN Train:
/usr/bin/python3 /home/mat/thesis/faster_rcnn_pytorch_multimodal/tools/trainval_net.py --iters 400000 --net res101 --db waymo --net_type lidar --iter 3 --scale 0.5 --preload 0 --en_full_net 0

Vanilla Net: (use RPN preload)
 /usr/bin/python3 /home/mat/thesis/faster_rcnn_pytorch_multimodal/tools/trainval_net.py --iters 400000 --net res101 --db waymo --net_type lidar --iter 1 --scale 0.5 --preload 2 --en_full_net 1 --weights_file ~/thesis/data/weights/lidar_rpn_50p_168k.pth 
 /usr/bin/python3 /home/mat/thesis/faster_rcnn_pytorch_multimodal/tools/trainval_net.py --iters 400000 --net res101 --db waymo --net_type lidar --iter 4 --scale 1.0 --preload 1 --en_full_net 1 --weights_file /home/mat/thesis/data/weights/res101_lidar_rpn_50p_168k.pth 

Aleatoric Only: (Use full preload)
 /usr/bin/python3 /home/mat/thesis/faster_rcnn_pytorch_multimodal/tools/trainval_net.py --iters 400000 --net res101 --db waymo --net_type lidar --iter 3 --scale 0.5 --preload 2 --en_full_net 1 --en_aleatoric 1 --uc_sort_type a_bbox_var --weights_file /home/mat/thesis/data/weights/res101_lidar_full_50p_36k.pth 

Epistemic Only:
/usr/bin/python3 /home/mat/thesis/faster_rcnn_pytorch_multimodal/tools/trainval_net.py --iters 400000 --net res101 --db waymo --net_type lidar --iter 3 --scale 0.5 --preload 1 --en_full_net 1 --en_epistemic 1 --uc_sort_type e_bbox_var --weights_file /home/mat/thesis/data/weights/res101_lidar_full_50p_36k.pth

Aleatoric & Epistemic: (Changed the dropout rate)
/usr/bin/python3 /home/mat/thesis/faster_rcnn_pytorch_multimodal/tools/trainval_net.py --iters 400000 --net res101 --db waymo --net_type lidar --iter 3 --scale 0.5 --preload 2 --en_full_net 1 --en_epistemic 1 --en_aleatoric 1 --weights_file /home/mat/thesis/data/weights/res101_lidar_full_50p_36k.pth 
----------------------------------------------------------------------------------
Image
Since its pretrained on caffe, only one command to run
python3 ./trainval_net.py --net_type image --weights_file ~/thesis/data/weights/res101-caffe.pth --train_iter 1 --preload 1

Current L2 mAP for vehicles: 0.51 (2D)
------------------------
Current sample commands:

RPN Train:
/usr/bin/python3 /home/mat/thesis/faster_rcnn_pytorch_multimodal/tools/trainval_net.py --iters 400000 --net res101 --db waymo --net_type lidar --iter 3 --scale 0.5 --preload 0 --en_full_net 0

Vanilla Net: (use RPN preload)
/usr/bin/python3 /home/mat/thesis/faster_rcnn_pytorch_multimodal/tools/trainval_net.py --iters 400000 --net res101 --db waymo --net_type image --iter 1 --scale 0.5 --preload 1 --en_full_net 1 --weights_file /home/mat/thesis/data/weights/res101_caffe.pth

Aleatoric Only:

Epistemic Only:

Aleatoric & Epistemic:
/usr/bin/python3 /home/mat/thesis/faster_rcnn_pytorch_multimodal/tools/trainval_net.py --iters 400000 --net res101 --db waymo --net_type image --iter 1 --scale 0.5 --preload 1 --en_full_net 1 --en_epistemic 1 --en_aleatoric 1 --uc_sort_type a_bbox_var --weights_file /home/mat/thesis/data/weights/res101_caffe.pth 
