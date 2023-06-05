'''
Created on Mar 1, 2023

@author: kurt
'''
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_FILE = os.path.abspath(__file__)

from time import gmtime, strftime
import logging
import matplotlib.pyplot as plt
import tensorflow as tf

# Init logging
format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")


class DigitKerasWriter(object):
    def __init__(self, name, log_dir, description=""):
        self.name=name
        self.log_dir=log_dir
        self.description=description
        
        '''
        tf.summary.create_file_writer(
            logdir,
            max_queue=None,
            flush_millis=None,
            filename_suffix=None,
            name=None,
            experimental_trackable=False)

        '''
        self.writer = tf.summary.create_file_writer(logdir=self.log_dir, name=self.name);
        logging.info("Writer created %s for dir %s "%(self.name, self.log_dir))
        
    def as_default(self): #DIGIT: 230301 Might be tricky
        return self.writer.as_default()
        
    def flush(self):
        self.writer.flush()
    def close(self):
        self.writer.flush()
        self.writer.close()    
    def reopen(self):
        self.writer.reopen()    
'''
----------------------------------------------------------------------------------------------------------------
'''        
            