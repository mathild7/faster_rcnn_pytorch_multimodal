# A simple generator wrapper, not sure if it's good for anything at all.
# With basic python threading
from threading import Thread
import time
from roi_data_layer.layer import RoIDataLayer
import sys
import numpy as np
#try:
#    from queue import Queue

#except ImportError:
#    from Queue import Queue
 
# ... or use multiprocessing versions
# WARNING: use sentinel based on value, not identity
import queue as ThreadQueue
from multiprocessing import Process, Queue, Value, Lock
import threading
from .config import cfg
#import multiprocessing

class data_layer_generator(object):
    """
    Generator that runs on a separate thread, returning values to calling
    thread. Care must be taken that the iterator does not mutate any shared
    variables referenced in the calling thread.
    """

    def __init__(self,
                 mode='train',
                 roidb=None,
                 augment_en=False,
                 num_classes=0,
                 Thread=Thread):

        if(mode == 'train'):
            self.data_layer = RoIDataLayer(roidb, num_classes, 'train')
        elif(mode == 'val'):
            self.data_layer = RoIDataLayer(roidb, num_classes, 'val', random=True)      
        self._queue = Queue(maxsize=16)
        #self._ptr_queue = Queue(maxsize=32)
        #self._perm_queue = Queue(maxsize=32)
        #self._v_queue = Queue(maxsize=32)
        self._daemon_en = Value('b',False)
        self._lock = Lock()
        self.finished   = False
        self._augment_en = Value('b',augment_en)
        self._cur = 0
        self._queue_count = 0
        self._perm = []
        if(cfg.DEBUG_EN):
            self._proc = threading.Thread(
                name='{} data generator'.format(mode),
                target=self._run,
                args=((self._lock, self._queue,self.data_layer,self._daemon_en, self._augment_en))
            )
        else:
            self._proc = Process(
                name='{} data generator'.format(mode),
                target=self._run,
                args=((self._lock, self._queue,self.data_layer,self._daemon_en, self._augment_en))
            )

    def __repr__(self):
        return 'ThreadedGenerator({!r})'.format(self._iterator)

    def is_queue_empty(self):
        print('queue is empty: {}'.format(self._queue.empty()))
        return self._queue.empty()

    def set_pointer(self,cur_val,perm_val):
        # current position in the validation database
        self.data_layer._cur = cur_val
        # current shuffled indexes of the validation database
        self.data_layer._perm = perm_val
        self._cur = cur_val
        self._perm = perm_val
        print('starting at {}, index: {} of array: {}'.format(self._cur,np.where(self._perm == self._cur),self._perm))

    def get_pointer(self):
        print('saving at {} {}'.format(self._cur,self._perm))
        return self._cur, self._perm

    def start(self):
        print('data generator starting up')
        with(self._lock):
            self._daemon_en.value = True
        self._proc.start()

    def kill(self):
        with(self._lock):
            print('killing data generator')
            self._daemon_en.value = False

    def _run(self, lock, q, data_layer, daemon_en, augment_en):
        while(True):
            with(lock):
                if(not daemon_en.value):
                    break
            if(not q.full()):
                blob = data_layer.forward(augment_en.value)
                #print('data acquired for image {}'.format(blob['filename']))
                q.put((blob, data_layer._cur, data_layer._perm))
                #self._ptr_queue.put(self.data_layer._cur)
                #self._perm_queue.put(self.data_layer._perm)
            else:
                time.sleep(0.01)
        #while(not q.empty()):
        #    print('queue not empty')
        #    time.sleep(1)
        
    def join(self):
        print(self._queue.empty())
        self._proc.join()

    def clear(self):
        try:
            while True:
                self._queue.get_nowait()
                #Love magic #'s
                time.sleep(0.1)
        except ThreadQueue.Empty:
            pass

    def next(self):
        while(self._queue.empty() and self._daemon_en is True):
            print('queue is empty, blocking..')
            time.sleep(0.05)
        #if(not self._queue.empty()):
        blobs, self._cur, self._perm = self._queue.get()
            #self._queue.task_done()
        return blobs
        #else:
            #print('shouldnt be here')
            #return None
