Faster-RCNN Image/LiDAR nets by Mat Hildebrand
Example run commands:
-----------------------------------------------------------------------------------
Lidar
First run
python3 ./trainval_net.py --net_type lidar --train_iter 1 --preload 0 --en_full_net 0

then after some training cycles run: (en_full_net is 1 by default)
python3 ./trainval_net.py --net_type lidar --weights_file ~/thesis/data/weights/lidar_rpn_60k.pth --train_iter 2 --preload 2

Current L2 mAP for vehicles: 0.3760
----------------------------------------------------------------------------------
Image
Since its pretrained on caffe, only one command to run
python3 ./trainval_net.py --net_type image --weights_file ~/thesis/data/weights/res101-caffe.pth --train_iter 1 --preload 1
