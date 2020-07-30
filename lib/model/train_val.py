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
from utils.filter_predictions import filter_and_draw_prep
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
from torchvision.ops import nms


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
                 db,
                 roidb,
                 val_roidb,
                 output_dir,
                 tbdir,
                 sum_size,
                 val_sum_size,
                 epoch_size,
                 batch_size,
                 val_batch_size,
                 val_thresh,
                 augment_en,
                 val_augment_en,
                 pretrained_model=None):
        self.net             = network
        self.db              = db
        self.roidb           = roidb
        self.val_roidb       = val_roidb
        self.output_dir      = output_dir
        self.tbdir           = tbdir
        self.sum_size        = sum_size
        self.val_sum_size    = val_sum_size
        self.epoch_size      = epoch_size
        self.batch_size      = batch_size
        self.val_batch_size  = val_batch_size
        self._val_augment_en = val_augment_en
        self._augment_en     = augment_en
        self.val_thresh      = val_thresh
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
        filename = cfg.NET_TYPE + '_' + cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}'.format(
            iter) + '.pth'
        filename = os.path.join(self.output_dir, filename)
        torch.save(self.net.state_dict(), filename)
        print('Wrote snapshot to: {:s}'.format(filename))

        # Also store some meta information, random state, etc.
        nfilename = cfg.NET_TYPE + '_' + cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}'.format(
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
        if(cfg.NET_TYPE == 'lidar'):
            #TODO: Magic numbers, need to sort this out to flow through 3d anchor gen properly
            self.net.create_architecture(
                self.db.num_classes,
                tag='default',
                anchor_scales=cfg.LIDAR.ANCHOR_SCALES,
                anchor_ratios=cfg.LIDAR.ANCHOR_ANGLES)
        elif(cfg.NET_TYPE == 'image'):
            self.net.create_architecture(
                self.db.num_classes,
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
        sfiles = os.path.join(self.output_dir,cfg.NET_TYPE + '_' +
                              cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_*.pth')
        sfiles = glob.glob(sfiles)
        sfiles.sort(key=os.path.getmtime)
        # Get the snapshot name in pytorch
        redfiles = []
        for stepsize in cfg.TRAIN.STEPSIZE:
            redfiles.append(
                os.path.join(
                    self.output_dir, cfg.NET_TYPE + '_' + cfg.TRAIN.SNAPSHOT_PREFIX +
                    '_iter_{:d}.pth'.format(stepsize + 1)))
        sfiles = [ss for ss in sfiles if ss not in redfiles]

        nfiles = os.path.join(self.output_dir,cfg.NET_TYPE + '_' +
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
        if(cfg.PRELOAD):
            print('Loading initial model weights from {:s}'.format(
                self.pretrained_model))
            self.net.load_pretrained_cnn(torch.load(self.pretrained_model))
        elif(cfg.PRELOAD_FULL):
            print('Loading initial full model weights {:s}'.format(
                self.pretrained_model))
            self.net.load_pretrained_full(torch.load(self.pretrained_model))
        else:
            print('initializing model from scratch')
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
        #self.data_layer = RoIDataLayer(self.roidb, self.db.num_classes, 'train')
        #self.data_layer_val = RoIDataLayer(self.val_roidb, self.db.num_classes, 'val', random=True)    
        self.data_gen     = data_layer_generator('train',self.roidb,self._augment_en,self.db.num_classes)
        self.data_gen_val = data_layer_generator('val',self.val_roidb,self._val_augment_en,self.db.num_classes)
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
        #TEST CODE TO TRY TO EXTRACT ONNX MODEL
        #self.net.eval()
        #dummy_input = torch.randn(1, 3, 1920, 730, device='cuda')
        #torch.onnx.export(self.net, dummy_input, '{}.onnx'.format(self.db.name), verbose=True)

        # Providing input and output names sets the display names for values
        # within the model's graph. Setting these does not change the semantics
        # of the graph; it is only for readability.
        #
        # The inputs to the network consist of the flat list of inputs (i.e.
        # the values you would pass to the forward() method) followed by the
        # flat list of parameters. You can partially specify names, i.e. provide
        # a list here shorter than the number of inputs to the model, and we will
        # only set that subset of names, starting from the beginning.
        t = utils.timer.Timer()
        timers = {}
        #timers['net'] = utils.timer.Timer()
        #timers['anchor_gen'] = utils.timer.Timer()
        #timers['proposal']   = utils.timer.Timer()
        #timers['proposal_t'] = utils.timer.Timer()
        #timers['anchor_t']   = utils.timer.Timer()
        timers['data_gen']   = utils.timer.Timer()
        #timers['losses']     = utils.timer.Timer()
        #timers['backprop']   = utils.timer.Timer()
        #timers['summary']     = utils.timer.Timer()
        self.net.timers = timers
        loss_cumsum = 0
        killer = GracefulKiller()
        if(iter < 10):
            self.db.delete_eval_draw_folder('trainval','train')
        #    scale_lr(self.optimizer,0.1)
        self.data_gen.start()
        self.data_gen_val.start()
        #Wait for samples to be pre-buffered
        time.sleep(3)
        while iter < max_iters + 1 and not killer.kill_now:

            #Start uncertainty capture late into training cycle
            #if(iter >= 5000 and cfg.UC.EN_BBOX_ALEATORIC is False):
            #    cfg.UC.EN_BBOX_ALEATORIC = True
            #    cfg.UC.EN_CLS_ALEATORIC  = True
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
            timers['data_gen'].tic()
            blobs  = self.data_gen.next()
            timers['data_gen'].toc()
            now = time.time()
            #if iter == 1  or now - last_summary_time > cfg.TRAIN.SUMMARY_INTERVAL:
            if iter % self.val_sum_size == 0:
                update_summaries = False
                for i in range(self.val_batch_size):
                    #blobs_val = self.data_layer_val.forward(val_augment_en)
                    blobs_val  = self.data_gen_val.next()
                    if(i == self.val_batch_size - 1):
                        update_summaries = True
                    if(cfg.UC.EN_BBOX_EPISTEMIC or cfg.UC.EN_CLS_EPISTEMIC):
                        self.net.set_e_num_sample(cfg.UC.E_NUM_SAMPLE)
                    summary_val, rois_val, roi_labels_val, \
                        cls_prob_val, bbox_pred_val, uncertainties_val = self.net.run_eval(blobs_val, self.val_batch_size, update_summaries)
                    if(cfg.UC.EN_BBOX_EPISTEMIC or cfg.UC.EN_CLS_EPISTEMIC):
                        self.net.set_e_num_sample(1)
                    #im info 0 -> H 1 -> W 2 -> scale
                    if(cfg.ENABLE_FULL_NET):
                        rois_val, bbox_pred_val, sorted_uncertainties_val = filter_and_draw_prep(rois_val, cls_prob_val,
                                                                                                    bbox_pred_val,
                                                                                                    uncertainties_val,
                                                                                                    blobs_val['info'],
                                                                                                    self.db.num_classes,self.val_thresh,self.db.type)
                        bbox_pred_val = np.array(bbox_pred_val)
                    #Final stage is stage 1, prep for drawing
                    else:
                        rois_val = rois_val.data.cpu().numpy()
                        roi_labels_val = roi_labels_val.data.cpu().numpy()
                        bbox_pred_val = bbox_pred_val
                        cls_prob_val  = cls_prob_val
                        keep = nms(bbox_pred_val, cls_prob_val.squeeze(1), cfg.TEST.NMS_THRESH).cpu().numpy() if cls_prob_val.shape[0] > 0 else []
                        bbox_pred_val = bbox_pred_val[keep,:].cpu().numpy()
                        cls_prob_val  = cls_prob_val[keep, :].cpu().numpy()
                        bbox_pred_val = np.concatenate((bbox_pred_val,cls_prob_val),axis=1)[np.newaxis,:,:]
                        bbox_pred_val = np.repeat(bbox_pred_val,2,axis=0)
                        sorted_uncertainties_val = [{},{}]

                    #Ensure that bbox_pred_val is a numpy array so that .size can be used on it.
                    #if(bbox_pred_val.size != 0):
                    #    bbox_pred_val = bbox_pred_val[:,:,:,np.newaxis]
                    self.db.draw_and_save_eval(blobs_val['filename'],rois_val,roi_labels_val,bbox_pred_val,sorted_uncertainties_val,iter+i,'train','trainval')

                #Need to add AP calculation here
                for _sum in summary_val:
                    self.valwriter.add_summary(_sum, float(iter))
            if iter % self.sum_size == 0:
                print('performing summary at iteration: {:d}'.format(iter))
                # Compute the graph with summary
                total_loss, summary = self.net.train_step_with_summary(blobs, self.optimizer, self.sum_size, update_weights)
                loss_cumsum += total_loss
                for _sum in summary:
                    #print('summary')
                    #print(_sum)
                    self.writer.add_summary(_sum, float(iter))
                last_summary_time = now
            else:
                # Compute the graph without summary
                total_loss = self.net.train_step(blobs, self.optimizer, update_weights)
                loss_cumsum += total_loss
            t.toc()

            # Display training information
            if (iter % cfg.TRAIN.DISPLAY) == 0:
                #self.net.print_cumulative_loss(iter-(iter%self.batch_size),iter, max_iters, lr)
                print('speed: {:.3f}s / iter'.format(
                    t.average_time()))
                for key, timer in timers.items():
                    print('{} timer: {:.3f}s / iter'.format(key,timer.average_time()))
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
              db,
              output_dir,
              tb_dir,
              pretrained_model=None,
              max_iters=40000,
              sum_size=128,
              val_sum_size=1000,
              batch_size=16,
              val_batch_size=16,
              val_thresh=0.1,
              augment_en=True,
              val_augment_en=False):
    """Train a Faster R-CNN network."""
    #roidb = filter_roidb(db.roidb)
    epoch_size = len(db.roidb)
    #val_roidb = filter_roidb(db.val_roidb)
    #TODO: merge with train_val as one entire object
    sw = SolverWrapper(
        network,
        db,
        db.roidb,
        db.val_roidb,
        output_dir,
        tb_dir,
        sum_size,
        val_sum_size,
        epoch_size,
        batch_size,
        val_batch_size,
        val_thresh,
        augment_en,
        val_augment_en,
        pretrained_model=pretrained_model)

    print('Solving...')
    sw.train_model(max_iters)
    print('done solving')
