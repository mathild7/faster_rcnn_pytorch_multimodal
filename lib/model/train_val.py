# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen and Zheqi He
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorboardX as tb
from model.bbox_transform import bbox_transform_inv, clip_boxes
from model.config import cfg
import roi_data_layer.roidb as rdl_roidb
from roi_data_layer.layer import RoIDataLayer
from .data_layer_generator import data_layer_generator
import utils.timer
try:
    import cPickle as pickle
except ImportError:
    import pickle

import torch
import torch.optim as optim
from torchvision.ops import nms
import numpy as np
import os
import sys
import glob
import time
import signal
import gc
from torch.multiprocessing import Pool, Process, Queue


class GracefulKiller:
  kill_now = False
  orig_sigint = None
  orig_sigterm = None
  def __init__(self):
    self.orig_sigint = signal.getsignal(signal.SIGINT)
    self.orig_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, self.exit_gracefully)
    signal.signal(signal.SIGTERM, self.exit_gracefully)

  def exit_gracefully(self,signum, frame):
    print('caught sigint/sigterm')
    signal.signal(signal.SIGINT, self.orig_sigint)
    signal.signal(signal.SIGTERM, self.orig_sigterm)
    self.kill_now = True




#TODO: Could use original imwidth/imheight
def compute_bbox(rois, cls_score, bbox_pred, bbox_var, imheight, imwidth, imscale, imdb,thresh=0.1):
    #print('validation img properties h: {} w: {} s: {} '.format(imheight,imwidth,imscale))
    rois = rois[:, 1:5] / imscale
    #Deleting extra dim
    #cls_score = np.reshape(cls_score, [cls_score.shape[0], -1])
    #bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])
    pred_boxes = bbox_transform_inv(torch.from_numpy(rois), torch.from_numpy(bbox_pred)).numpy()
    # x1 >= 0
    pred_boxes[:, 0::4] = np.maximum(pred_boxes[:, 0::4], 0)
    # y1 >= 0
    pred_boxes[:, 1::4] = np.maximum(pred_boxes[:, 1::4], 0)
    # x2 < imwidth
    pred_boxes[:, 2::4] = np.minimum(pred_boxes[:, 2::4], imwidth/imscale - 1)
    # y2 < imheight 3::4 means start at 3 then jump every 4
    pred_boxes[:, 3::4] = np.minimum(pred_boxes[:, 3::4], imheight/imscale - 1)
    all_boxes = []
    # skip j = 0, because it's the background class
    #for i,score in enumerate(cls_score):
    #    for j,cls_s in enumerate(score):
    #        if((cls_s > thresh or cls_s < 0.0) and j > 0):
                #print('score for entry {} and class {} is {}'.format(i,j,cls_s))

    for j in range(1, imdb.num_classes):
        #Don't need to stack variance here, it is only used in trainval to draw stuff
        all_box, _ = imdb.nms_hstack(cls_score,pred_boxes,np.empty(0),bbox_var,thresh,j,stack_var=False)
        all_boxes.append(all_box)
    return rois, all_boxes

def scale_lr(optimizer, scale):
    """Scale the learning rate of the optimizer"""
    for param_group in optimizer.param_groups:
        param_group['lr'] *= scale

class SolverWrapper(object):
    """
    A wrapper class for the training process
  """

    def __init__(self,
                 network,
                 imdb,
                 roidb,
                 val_roidb,
                 output_dir,
                 tbdir,
                 sum_size,
                 val_sum_size,
                 epoch_size,
                 batch_size,
                 val_im_thresh,
                 augment_en,
                 val_augment_en,
                 pretrained_model=None):
        self.net = network
        self.imdb = imdb
        self.roidb = roidb
        self.val_roidb = val_roidb
        self.output_dir = output_dir
        self.tbdir = tbdir
        self.sum_size = sum_size
        self.val_sum_size = val_sum_size
        self.epoch_size = epoch_size
        self.batch_size = batch_size
        self.val_batch_size = batch_size
        self._val_augment_en = val_augment_en
        self._augment_en     = augment_en
        self.val_im_thresh = val_im_thresh
        # Simply put '_val' at the end to save the summaries from the validation set
        self.tbvaldir = tbdir + '_val'
        if not os.path.exists(self.tbvaldir):
            os.makedirs(self.tbvaldir)
        self.pretrained_model = pretrained_model

    def snapshot(self, iter):
        net = self.net

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Store the model snapshot
        filename = cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}'.format(
            iter) + '.pth'
        filename = os.path.join(self.output_dir, filename)
        torch.save(self.net.state_dict(), filename)
        print('Wrote snapshot to: {:s}'.format(filename))

        # Also store some meta information, random state, etc.
        nfilename = cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}'.format(
            iter) + '.pkl'
        nfilename = os.path.join(self.output_dir, nfilename)
        # current state of numpy random
        st0 = np.random.get_state()
        # current position in the database
        #cur = self.data_layer._cur
        # current shuffled indexes of the database
        #perm = self.data_layer._perm
        # current position in the validation database
        #cur_val  = self.data_layer_val._cur
        # current shuffled indexes of the validation database
        #perm_val = self.data_layer_val._perm

        cur_val, perm_val = self.data_gen_val.get_pointer()
        cur, perm         = self.data_gen.get_pointer()

        # Dump the meta info
        with open(nfilename, 'wb') as fid:
            pickle.dump(st0, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(cur, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(perm, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(cur_val, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(perm_val, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(iter, fid, pickle.HIGHEST_PROTOCOL)

        return filename, nfilename

    def from_snapshot(self, sfile, nfile):
        print('Restoring model snapshots from {:s}'.format(sfile))
        self.net.load_state_dict(torch.load(str(sfile)))
        print('Restored.')
        # Needs to restore the other hyper-parameters/states for training, (TODO xinlei) I have
        # tried my best to find the random states so that it can be recovered exactly
        # However the Tensorflow state is currently not available
        with open(nfile, 'rb') as fid:
            st0 = pickle.load(fid)
            cur = pickle.load(fid)
            perm = pickle.load(fid)
            cur_val = pickle.load(fid)
            perm_val = pickle.load(fid)
            last_snapshot_iter = pickle.load(fid)

            np.random.set_state(st0)
            #self.data_layer._cur = cur
            #self.data_layer._perm = perm
            #self.data_layer_val._cur = cur_val
            #self.data_layer_val._perm = perm_val
            self.data_gen.set_pointer(cur,perm)
            self.data_gen_val.set_pointer(cur_val,perm_val)

        return last_snapshot_iter

    def construct_graph(self):
        # Set the random seed
        torch.manual_seed(cfg.RNG_SEED)
        # Build the main computation graph
        self.net.create_architecture(
            self.imdb.num_classes,
            tag='default',
            anchor_scales=cfg.ANCHOR_SCALES,
            anchor_ratios=cfg.ANCHOR_RATIOS)
        # Define the loss
        # loss = layers['total_loss']
        # Set learning rate and momentum
        lr = cfg.TRAIN.LEARNING_RATE
        params = []
        for key, value in dict(self.net.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{
                        'params': [value],
                        'lr':
                        lr * (cfg.TRAIN.DOUBLE_BIAS + 1),
                        'weight_decay':
                        cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0
                    }]
                else:
                    params += [{
                        'params': [value],
                        'lr':
                        lr,
                        'weight_decay':
                        getattr(value, 'weight_decay', cfg.TRAIN.WEIGHT_DECAY)
                    }]
        #self.optimizer = torch.optim.Adam(params)
        self.optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)
        # Write the train and validation information to tensorboard
        self.writer = tb.writer.FileWriter(self.tbdir)
        self.valwriter = tb.writer.FileWriter(self.tbvaldir)

        return lr, self.optimizer

    def find_previous(self):
        sfiles = os.path.join(self.output_dir,
                              cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_*.pth')
        sfiles = glob.glob(sfiles)
        sfiles.sort(key=os.path.getmtime)
        # Get the snapshot name in pytorch
        redfiles = []
        for stepsize in cfg.TRAIN.STEPSIZE:
            redfiles.append(
                os.path.join(
                    self.output_dir, cfg.TRAIN.SNAPSHOT_PREFIX +
                    '_iter_{:d}.pth'.format(stepsize + 1)))
        sfiles = [ss for ss in sfiles if ss not in redfiles]

        nfiles = os.path.join(self.output_dir,
                              cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_*.pkl')
        nfiles = glob.glob(nfiles)
        nfiles.sort(key=os.path.getmtime)
        redfiles = [redfile.replace('.pth', '.pkl') for redfile in redfiles]
        nfiles = [nn for nn in nfiles if nn not in redfiles]

        lsf = len(sfiles)
        assert len(nfiles) == lsf

        return lsf, nfiles, sfiles

    def initialize(self):
        # Initial file lists are empty
        np_paths = []
        ss_paths = []
        # Fresh train directly from ImageNet weights
        print('Loading initial model weights from {:s}'.format(
            self.pretrained_model))
        self.net.load_pretrained_cnn(torch.load(self.pretrained_model))
        #self.net.load_trimmed_pretrained_cnn(torch.load(self.pretrained_model))
        #sys.exit('test ended')
        print('Loaded.')
        # Need to fix the variables before loading, so that the RGB weights are changed to BGR
        # For VGG16 it also changes the convolutional weights fc6 and fc7 to
        # fully connected weights
        last_snapshot_iter = 0
        lr = cfg.TRAIN.LEARNING_RATE
        stepsizes = list(cfg.TRAIN.STEPSIZE)

        return lr, last_snapshot_iter, stepsizes, np_paths, ss_paths

    def restore(self, sfile, nfile):
        # Get the most recent snapshot and restore
        np_paths = [nfile]
        ss_paths = [sfile]
        # Restore model from snapshots
        last_snapshot_iter = self.from_snapshot(sfile, nfile)
        # Set the learning rate
        lr_scale = 1
        stepsizes = []
        for stepsize in cfg.TRAIN.STEPSIZE:
            if last_snapshot_iter > stepsize:
                #lr_scale = cfg.TRAIN.GAMMA*lr_scale
                print('pre reducing learning rate')
                lr_scale *= cfg.TRAIN.GAMMA
            else:
                stepsizes.append(stepsize)
        scale_lr(self.optimizer, lr_scale)
        lr = cfg.TRAIN.LEARNING_RATE * lr_scale
        return lr, last_snapshot_iter, stepsizes, np_paths, ss_paths

    def remove_snapshot(self, np_paths, ss_paths):
        to_remove = len(np_paths) - cfg.TRAIN.SNAPSHOT_KEPT
        for c in range(to_remove):
            nfile = np_paths[0]
            print('removing snapshot file {:s}'.format(nfile))
            os.remove(str(nfile))
            np_paths.remove(nfile)

        to_remove = len(ss_paths) - cfg.TRAIN.SNAPSHOT_KEPT
        for c in range(to_remove):
            sfile = ss_paths[0]
            # To make the code compatible to earlier versions of Tensorflow,
            # where the naming tradition for checkpoints are different
            os.remove(str(sfile))
            ss_paths.remove(sfile)

    def train_model(self, max_iters):
        # Build data layers for both training and validation set
        update_weights = False
        val_augment_en = self._val_augment_en
        augment_en     = self._augment_en
        val_batch_size = self.batch_size
        #self.data_layer = RoIDataLayer(self.roidb, self.imdb.num_classes, 'train')
        #self.data_layer_val = RoIDataLayer(self.val_roidb, self.imdb.num_classes, 'val', random=True)    
        self.data_gen     = data_layer_generator('train',self.roidb,augment_en,self.imdb.num_classes)
        self.data_gen_val = data_layer_generator('val',self.val_roidb,val_augment_en,self.imdb.num_classes)
        # Construct the computation graph
        lr, train_op = self.construct_graph()

        # Find previous snapshots if there is any to restore from
        lsf, nfiles, sfiles = self.find_previous()

        # Initialize the variables or restore them from the last snapshot
        if lsf == 0:
            lr, last_snapshot_iter, stepsizes, np_paths, ss_paths = self.initialize(
            )
        else:
            lr, last_snapshot_iter, stepsizes, np_paths, ss_paths = self.restore(
                str(sfiles[-1]), str(nfiles[-1]))
        iter = last_snapshot_iter + 1
        last_summary_time = time.time()
        # Make sure the lists are not empty
        #Append step sizes (in this case epoch) to list
        stepsizes.append(max_iters)
        stepsizes.reverse()
        #print(stepsizes)
        next_stepsize = stepsizes.pop()
        #print(stepsizes)
        #Switch to train mode
        self.net.train()
        self.net.to(self.net._device)
        #self.net.eval()
        #dummy_input = torch.randn(1, 3, 1920, 730, device='cuda')
        #torch.onnx.export(self.net, dummy_input, '{}.onnx'.format(self.imdb.name), verbose=True)
# Providing input and output names sets the display names for values
# within the model's graph. Setting these does not change the semantics
# of the graph; it is only for readability.
#
# The inputs to the network consist of the flat list of inputs (i.e.
# the values you would pass to the forward() method) followed by the
# flat list of parameters. You can partially specify names, i.e. provide
# a list here shorter than the number of inputs to the model, and we will
# only set that subset of names, starting from the beginning.
        t = utils.timer.timer
        t_net = utils.timer.timer
        loss_cumsum = 0
        killer = GracefulKiller()
        if(iter < 10):
            self.imdb.delete_eval_draw_folder('val','trainval')
        #    scale_lr(self.optimizer,0.1)
        self.data_gen.start()
        self.data_gen_val.start()
        time.sleep(3)
        while iter < max_iters + 1 and not killer.kill_now:

            #if(iter >= 20000 and cfg.ENABLE_BBOX_VAR is False):
            #    cfg.ENABLE_BBOX_VAR = True
            #print('iteration # {}'.format(iter))
            # Learning rate
            if iter % self.batch_size == 0 and iter != 0:
                update_weights = True
            else:
                update_weights = False
            if iter == next_stepsize + 1:
                # Add snapshot here before reducing the learning rate
                self.snapshot(iter)
                #change learning rate every step size
                print('Reducing learning rate')
                lr *= cfg.TRAIN.GAMMA
                scale_lr(self.optimizer, cfg.TRAIN.GAMMA)
                next_stepsize = stepsizes.pop()
            #if iter == 500:
                #print('bumping learning rate back up')
                #scale_lr(self.optimizer,10)
            t.tic()
            # Get training data, one batch at a time
            #blobs = self.data_layer.forward(augment_en)
            blobs  = self.data_gen.next()
            now = time.time()
            #if iter == 1  or now - last_summary_time > cfg.TRAIN.SUMMARY_INTERVAL:
            if iter % self.val_sum_size == 0:
                update_summaries = False
                for i in range(val_batch_size):
                    #blobs_val = self.data_layer_val.forward(val_augment_en)
                    blobs_val  = self.data_gen_val.next()
                    if(i == val_batch_size - 1):
                        update_summaries = True
                    summary_val, rois_val, roi_labels_val, bbox_pred_val, bbox_var_val, cls_prob_val = self.net.run_eval(blobs_val, val_batch_size, update_summaries)
                    #im info 0 -> H 1 -> W 2 -> scale
                    #Add ability to do compute_bbox on rois_val and pass to draw&save
                    rois_val, bbox_pred_val = compute_bbox(rois_val,cls_prob_val,bbox_pred_val,bbox_var_val,blobs_val['im_info'][0],blobs_val['im_info'][1],blobs_val['im_info'][2], self.imdb,self.val_im_thresh)
                    bbox_pred_val = np.array(bbox_pred_val)
                    bbox_pred_val = bbox_pred_val[:,:,:,np.newaxis]
                    self.imdb.draw_and_save_eval(blobs_val['imagefile'],rois_val,roi_labels_val,bbox_pred_val,iter+i,'trainval')

                    #for obj in gc.get_objects():
                    #    try:
                    #        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    #            print(type(obj), obj.size())
                    #    except:
                    #        pass
                #Need to add AP calculation here
                for _sum in summary_val:
                    self.valwriter.add_summary(_sum, float(iter))

            if iter % self.sum_size == 0:
                #print('performing summary at iteration: {:d}'.format(iter))
                # Compute the graph with summary
                rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, total_loss, summary = \
                  self.net.train_step_with_summary(blobs, self.optimizer, self.sum_size, update_weights)
                loss_cumsum += total_loss
                for _sum in summary:
                    #print('summary')
                    #print(_sum)
                    self.writer.add_summary(_sum, float(iter))
                last_summary_time = now
            else:
                # Compute the graph without summary
                #https://stackoverflow.com/questions/46561390/4-step-alternating-rpn-faster-r-cnn-training-tensorflow-object-detection-mo/46981671#46981671
                #https://arthurdouillard.com/post/faster-rcnn/
                #Alternate sharing: 
                    #Similar to some matrix decomposition methods, the authors train RPN, then Fast-RCN, and so on. Each network is trained a bit alternatively.
                #Approximate joint training: 
                    #This strategy consider the two networks as a single unified one. The back-propagation uses both the Fast-RCNN loss and the RPN loss. However the regression of bounding-box coordinates in RPN is considered as pre-computed, and thus its derivative is ignored.
                #Non-approximate joint training:
                    #This solution was not used as more difficult to implement. The RoI pooling is made differentiable w.r.t the box coordinates using a RoI warping layer.
                #4-Step Alternating training: 
                    #The strategy chosen takes 4 steps: In the first of one the RPN is trained. In the second, Fast-RCNN is trained using pre-computed RPN proposals. For the third step, the trained Fast-RCNN is used to initialize a new RPN where only RPN’s layers are fine-tuned. Finally in the fourth step RPN’s layers are frozen and only Fast-RCNN is fine-tuned.
                rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, total_loss = \
                  self.net.train_step(blobs, self.optimizer, update_weights)
                loss_cumsum += total_loss
            t.toc()

            # Display training information
            if (iter % cfg.TRAIN.DISPLAY) == 0:
                self.net.print_cumulative_loss(iter-(iter%self.batch_size),iter, max_iters, lr)
                print('speed: {:.3f}s / iter'.format(
                    t.average_time()))
            if iter % self.epoch_size == 0:
                print('----------------------------------------------------')
                print('epoch average loss: {:f}'.format(float(loss_cumsum)/float(self.epoch_size)))
                print('----------------------------------------------------')
                loss_cumsum = 0
                # for k in utils.timer.timer._average_time.keys():
                #   print(k, utils.timer.timer.average_time(k))

            # Snapshotting
            if iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = iter
                ss_path, np_path = self.snapshot(iter)
                np_paths.append(np_path)
                ss_paths.append(ss_path)

                # Remove the old snapshots if there are too many
                if len(np_paths) > cfg.TRAIN.SNAPSHOT_KEPT:
                    self.remove_snapshot(np_paths, ss_paths)
            iter += 1
        print('killing data generators')
        self.data_gen.kill()
        self.data_gen_val.kill()
        time.sleep(1)
        self.data_gen.clear()
        self.data_gen_val.clear()
        #while(not self.data_gen.is_queue_empty()):
        #    print('emptying queue value')
        #    dummy_var = self.data_gen.next()
        #while(not self.data_gen_val.is_queue_empty()):
        #    print('emptying queue value')
        #    dummy_var = self.data_gen_val.next()
        self.data_gen.join()
        self.data_gen_val.join()
        if(killer.kill_now):
            print("Killing program")
            sys.exit()
        if last_snapshot_iter != iter - 1:
            self.snapshot(iter - 1)

        self.writer.close()
        self.valwriter.close()

def filter_roidb(roidb):
    """Remove roidb entries that have no usable RoIs."""
    def is_valid(entry):
        # Valid images have:
        #   (1) At least one foreground RoI OR
        #   (2) At least one background RoI
        if('max_overlaps' not in entry):
            print('about to fail')
            print(entry)
        overlaps = entry['max_overlaps']
        # find boxes with sufficient overlap
        fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                           (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
        # image is only valid if such boxes exist
        valid = len(fg_inds) > 0 or len(bg_inds) > 0
        return valid

    num = len(roidb)
    filtered_roidb = [entry for entry in roidb if is_valid(entry)]
    num_after = len(filtered_roidb)
    print('Filtered {} roidb entries: {} -> {}'.format(num - num_after, num,
                                                       num_after))
    return filtered_roidb


def train_net(network,
              imdb,
              output_dir,
              tb_dir,
              pretrained_model=None,
              max_iters=40000,
              sum_size=128,
              val_sum_size=1000,
              batch_size=16,
              val_im_thresh=0.1,
              augment_en=True,
              val_augment_en=False):
    """Train a Faster R-CNN network."""
    #roidb = filter_roidb(imdb.roidb)
    epoch_size = len(imdb.roidb)
    #val_roidb = filter_roidb(imdb.val_roidb)
    #TODO: merge with train_val as one entire object
    sw = SolverWrapper(
        network,
        imdb,
        imdb.roidb,
        imdb.val_roidb,
        output_dir,
        tb_dir,
        sum_size,
        val_sum_size,
        epoch_size,
        batch_size,
        val_im_thresh,
        augment_en,
        val_augment_en,
        pretrained_model=pretrained_model)

    print('Solving...')
    sw.train_model(max_iters)
    print('done solving')
