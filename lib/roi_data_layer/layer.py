# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
"""The data layer used during training to train a Fast R-CNN network.

RoIDataLayer implements a Caffe Python layer.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.config import cfg
from roi_data_layer.minibatch import get_minibatch
import numpy as np
import time


class RoIDataLayer(object):
    """Fast R-CNN data layer used for training."""
    #TODO: Handle validation and training seperately.
    def __init__(self, roidb, num_classes, mode, random=False):
        """Set the roidb to be used by this layer during training."""
        self._roidb = roidb
        self._mode  = mode
        self._num_classes = num_classes
        # Also set a random flag
        self._random = random
        self._shuffle_roidb_inds()

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        # If the random flag is set,
        # then the database is shuffled according to system time
        # Useful for the validation set
        if self._random:
            st0 = np.random.get_state()
            millis = int(round(time.time() * 1000)) % 4294967295
            np.random.seed(millis)

        self._perm = np.random.permutation(np.arange(len(self._roidb)))
        # Restore the random state
        if self._random:
            np.random.set_state(st0)
        #Reset random shuffle back to 0
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        #TODO: Should we really be shuffling once roidb is rolled over? (for training at least)
        if self._cur + cfg.TRAIN.FRAMES_PER_BATCH >= len(self._roidb):
            print('shuffling indices')
            self._shuffle_roidb_inds()
        if(cfg.FREEZE_DB):
            self._cur = cfg.FREEZE_DB_INDS
        db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.FRAMES_PER_BATCH]
        self._cur += cfg.TRAIN.FRAMES_PER_BATCH
        #print('getting db inds {}'.format(db_inds))
        return db_inds

    def _get_next_minibatch(self, augment_en):
        """Return the blobs to be used for the next minibatch.

    If cfg.TRAIN.USE_PREFETCH is True, then blobs will be computed in a
    separate process and made available through self._blob_queue.
    """
        minibatch = None
        while (minibatch is None):
            db_inds = self._get_next_minibatch_inds()
            minibatch_db = [self._roidb[i] for i in db_inds]
            #print('minibatch')
            #print(minibatch_db)
            minibatch = get_minibatch(minibatch_db, self._num_classes, augment_en)
            #if(minibatch is None):
                #print('skipping image, augmentation resulted in 0 GT boxes')
        return minibatch

    def forward(self,augment_en):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch(augment_en)
        return blobs
