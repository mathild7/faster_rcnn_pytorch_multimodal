Analyzing variance commit:
common:
- Changed 'img_index' to 'imgname' in each ROI
imdb.py:
- Added nms_hstack() function, publically accessible. This replaces the nms operation done twice in the code in different places
- Added 'image_index_at' which is a function that finds the specific image identifier (scene*1000+img) based on the shuffled index
nuscenes_imdb.py:
- _write_nuscenes_results_file function: Added variance output to detection file generator
waymo_eval.py:
- added a variable passthrough to take cfg.TEST.TOD_FILTER_LIST or cfg.TRAIN.TOD_FILTER_LIST to the ROI generation function
- Modified alert message to inform the user if a specific image has no entry in the ROIDB or if it has no GT boxes
- Added BB_var variable that extracts each detection variance from the output file.
- Added logic to accumulate variance on a per scene and per image basis[scene][img]. This is then used later to compute variances
- Added print function to print the average variance across a scene.
waymo_imdb.py:
- Added tod_filter_list as an initialization arg, this can be used to filter out certain scenes, Day,Night,Dawn/Dusk. ATTN: This will not be utilized UNLESS the roidb cache is removed
- Added scene_from_index() Doesn't do anything yet, probably can remove
- Modified _draw_and_save_eval() to handle multiple boxes per detection. Generally this means extending out the numpy array of class detection scores to an extra dimension
- Modified _load_waymo_annotation() to include a tod_filter_list. This way  when loading ROI's from the labels.json, a non magic list will specify from which scenes to pull.
- Modified ROI dict, to include image_index, scene_index, scene_description and moved the image filename variable to entry 'imgname'
- Modified _write_waymo_results_file() to include BBOX variance measure in its output detection file generation

proposal_target_layer.py:
- Added comments to confirm that bbox_target_data is indeed the delta measurement pre-transform.

config.py:
- Added several new variables to better control variance

data_layer_generator.py:
- Added functionality to enable debug in train mode, processes do not debug, threads do.

test.py:
- modified im_detect(): Handled an additional bbox_var output from the test_image function call. This is also where the bounding boxes are currently sampled from the mean and variance. Need to move num_samples to a cfg parameter
- modified im_detect(): pred_boxes are now generated one sample at a time for each detection.
- modified test_net(): Moved NMS functionality to the imdb class, even though it probably should be somewhere else. nms_hstack performs trimming of detections and horizontally stacking of three elements (bbox_pred,bbox_var,cls_prob) on a per class basis.
- modified test_net(): Added additional functionality to get just the (bbox_pred,cls_prob) horizontal stack. Can actually probably re-call nms_stack may be cleaner


train_val.py:
- Modified compute_bbox(): changed the signature of the function to take in the imdb, instead of num_classes. 
- Modified compute_bbox(): Removed hstack/nms functionality and ported it to the imdb.py class.
- Modified solver_wrapper class: Added variance output from run_eval() call, also added bbox_var to compute_bbox call, however this is actually dropped in compute_bbox, as stack_var is False


network.py:
- modified smooth_l1_loss(): Included ability to use bbox_var in the loss
- modified smooth_l1_loss(): Added stage discrimination by adding a variable in the signature, determine if RPN or 2nd stage
- Added _bbox_mean_entropy(): Unused currently, may have a better use for it yet.
- Added functionality to extract bbox_variance from an additional fully connected layer.
- Added normal initialization of bbox_var_net mean:0 var:0.001 mode: truncated
- Added average variance of each sample to self._losses, so it can be tracked on tensorboard







-------------------------------------------------------
pedestrian
cyclist
car

70000 iterations
~~~~~~~
Results:
E 0.842 M 0.726 H 0.662
E 0.973 M 0.910 H 0.776
E 0.989 M 0.719 H 0.653
Mean AP = E 0.9347 M 0.7849 H 0.6972
~~~~~~~~

210001 iterations
~~~~~~~~
Results:
E 0.864 M 0.776 H 0.703
E 0.964 M 0.902 H 0.783
E 0.997 M 0.922 H 0.820
Mean AP = E 0.9417 M 0.8663 H 0.7688
~~~~~~~~

320000 iterations
~~~~~~~~
Results:
E 0.870 M 0.742 H 0.680
E 0.982 M 0.931 H 0.807
E 0.993 M 0.921 H 0.812
Mean AP = E 0.9485 M 0.8645 H 0.7664
~~~~~~~~
