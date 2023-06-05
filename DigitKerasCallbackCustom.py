'''
Created on 28 May 2021

@author: digit
'''
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import tensorflow as tf

class DigitKerasCallbackCustom(tf.keras.callbacks.Callback):
    
    def __init__(self, dirname, modelname, log_dir, verbose=False):
        
        self.dirname   = dirname
        self.modelname = modelname
        self.log_dir = log_dir
        
        self.verbose = True#verbose
        super().__init__()
    
    def on_train_begin(self, logs=None):
        if logs is None: return
        keys = list(logs.keys())
        if self.verbose: print("Starting training; got log keys: {}".format(keys))

    def on_train_end(self, logs=None):
        if logs is None: return
        keys = list(logs.keys())
        if self.verbose: print("Stop training; got log keys: {}".format(keys))

    def on_epoch_begin(self, epoch, logs=None):
        if logs is None: return
        keys = list(logs.keys())
        if self.verbose: print("Start epoch {} of training; got log keys: {}".format(epoch, keys))

    def on_epoch_end(self, epoch, logs=None):
        if logs is None: return
        keys = list(logs.keys())
        #End epoch 0 of training; got log keys: ['loss', 'acc', 'val_loss', 'val_acc', 'lr']
        print("End epoch {} of training; got log keys: {}".format(epoch, keys))
        if self.verbose: print("End epoch {} of training; got log keys: {}".format(epoch, keys))

    def on_test_begin(self, logs=None):
        if logs is None: return
        keys = list(logs.keys())
        if self.verbose: print("Start testing; got log keys: {}".format(keys))

    def on_test_end(self, logs=None):
        if logs is None: return
        keys = list(logs.keys())
        if self.verbose: print("Stop testing; got log keys: {}".format(keys))

    def on_predict_begin(self, logs=None):
        if logs is None: return
        keys = list(logs.keys())
        if self.verbose: print("Start predicting; got log keys: {}".format(keys))

    def on_predict_end(self, logs=None):
        if logs is None: return
        keys = list(logs.keys())
        if self.verbose: print("Stop predicting; got log keys: {}".format(keys))

    def on_train_batch_begin(self, batch, logs=None):
        if logs is None: return
        keys = list(logs.keys())
        if self.verbose: print("...Training: start of batch {}; got log keys: {}".format(batch, keys))

    def on_train_batch_end(self, batch, logs=None):
        if logs is None: return
        keys = list(logs.keys())
        if self.verbose: print("...Training: end of batch {}; got log keys: {}".format(batch, keys))

    def on_test_batch_begin(self, batch, logs=None):
        if logs is None: return
        keys = list(logs.keys())
        if self.verbose: print("...Evaluating: start of batch {}; got log keys: {}".format(batch, keys))

    def on_test_batch_end(self, batch, logs=None):
        if logs is None: return
        keys = list(logs.keys())
        if self.verbose: print("...Evaluating: end of batch {}; got log keys: {}".format(batch, keys))

    def on_predict_batch_begin(self, batch, logs=None):
        if logs is None: return
        keys = list(logs.keys())
        if self.verbose: print("...Predicting: start of batch {}; got log keys: {}".format(batch, keys))

    def on_predict_batch_end(self, batch, logs=None):
        if logs is None: return
        keys = list(logs.keys())
        if self.verbose: print("...Predicting: end of batch {}; got log keys: {}".format(batch, keys))


'''
-----------------------------------------------------
''' 
if __name__ == '__main__':  
    pass
'''
-----------------------------------------------------
''' 
    
    

