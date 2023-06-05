'''
Created on 31.01.2021

@author: DIGIT
'''
import os
# import matplotlib.pyplot as plt
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
BASE_FILE = os.path.abspath(__file__)

# from DigitNvidia import set_gpu#CPU/GPU
# set_gpu(gpu=True, set_memory=True)

# import gc
import numpy as np
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
tf.get_logger().setLevel('INFO')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy

# own imports
# -----------
from BM_ModelData_dkube import BM_ModelData
from DigitKerasModel_dkube import DigitKerasModel

import DigitConfig
''' Config Json File '''
conf = DigitConfig.Config('_BostonMedical')

# import Model

from  BM_DenseB_008_2D import BM_DenseB_008_2D

class BM_Train_DenseB008(DigitKerasModel):
    
    def __init__(self, 
                 # data_paths
                 # ----------
                 img_dir,
                 mask_dir,
                 res_dir,
                 # input data
                 img_size = (160, 160),
                 num_channels=1,
                 num_slices=None, 
                 # model_data
                 num_classes = 3,
                 # input cols for hybrid
                 input_cols=1,#8,  
                 # use data gen for training
                 # -------------------------
                 use_data_gen = True,
                 data_augment = False,
                 # architecture
                 network_ID=99,
                 # network details
                 # ---------------
                 growth_rate=8,
                 drop_rate=0.2,
                 use_trans=False,
                 filters=None,
                 # training
                 # --------
                 epochs = 12,
                 batch_size = 32,
                 # training details
                 max_samples= 800,
                 max_epochs = 2,
                 test_split_ratio = 0.2,
                 loss_weights=None,
                 
                 # paths and names for model
                 # -------------------------
                 dirname=None, 
                 modelname=None):
        
        # data_paths
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.res_dir = res_dir
        # input data
        self.img_size = img_size
        self.num_channels = num_channels
        self.num_slices = num_slices
        
        # model data
        self.num_classes = num_classes
        
        # input cols for hybrid
        self.num_slices = num_slices
        
        
        # use data gen for training
        # -------------------------
        self.use_data_gen = use_data_gen
        
        use_data_gen = True,
        self.data_augment = data_augment
        
        # architecture
        # ------------
        self.network_ID=network_ID
        
        # network details
        # ---------------
        self.growth_rate=growth_rate
        self.drop_rate=drop_rate
        self.use_trans=use_trans
        self.filters=filters
        
        # training
        # --------
        self.epochs=epochs
        self.batch_size=batch_size
        
        # training details
        self.max_samples = max_samples
        self.max_epochs=max_epochs
        self.test_split_ratio = test_split_ratio
        self.loss_weights = np.asarray(loss_weights) if loss_weights is not None else None
                 
        # init class members
        # ------------------
        self.model = None
        self.callbacks = None
        self.history = None
        self.pred = None
        self.initial_epoch=None

        # init input x, output y
        # ----------------------
        (self.x_train, self.y_train, self.x_test,  self.y_test) = (None, None, None, None)
        (self.train_gen, self.test_gen, self.val_gen, self.all_gen ) = (None, None, None, None)
        (self.num_train, self.num_test)  = (0,0) #Digit: obsolete
        
        # paths and name of model
        self.dirname = dirname or os.path.join(BASE_DIR,"_architecture")
        
        self.modelname = modelname or type(self).__name__ 
        print("modelname ",self.modelname)
        print("*"*200)
        
        # init base class new in Python 3.0
        super().__init__(dirname=self.dirname, modelname=self.modelname)
        #DigitKerasModel.__init__(self, dirname=self.dirname, modelname=self.modelname)

        
    
            
    def get_data_gen(self):
        
        print(self.modelname, " ", "get_data")
        c_data=BM_ModelData(img_dir  = self.img_dir,
                            mask_dir = self.mask_dir, 
                            batch_size=self.batch_size, 
                            img_size=self.img_size,
                            num_channels=self.num_channels,
                            max_samples=self.max_samples,
                            test_split_ratio=self.test_split_ratio,
                            data_augment = self.data_augment)
        
        train_gen, test_gen, val_gen = c_data.get_data()
        
        print("UN_MODEL train_gen %d batches"% len(train_gen))
        print("UN_MODEL test_gen  %d batches"% len(test_gen))
        print("UN_MODEL val_gen   %d batches"% len(val_gen))
        
        #assert len(test_gen) == len(val_gen)
        
        self.train_gen = train_gen 
        self.test_gen  = test_gen
        self.val_gen   = val_gen
        return train_gen, test_gen, val_gen
    
    def get_all_gen(self, max_samples=None):    
        c_data_all=BM_ModelData(img_dir  = self.img_dir,
                            mask_dir = self.mask_dir, 
                            batch_size=batch_size, 
                            #------------
                            img_size=self.img_size,
                            num_channels=self.num_channels,
                            max_samples=max_samples,
                            shuffle=False,
                            test_split_ratio=self.test_split_ratio)
        
        
        self.all_gen=c_data_all.get_all_data(max_samples=None)
        print("UN_MODEL all_gen   %d"% len(self.all_gen))
        return self.all_gen
        
    
'''
-----------------------------------------------------
'''    
if __name__ == '__main__':  
    # Free up RAM in case the model definition cells were run multiple times
    tf.keras.backend.clear_session()
  
    try:
        modelname = conf["MODELNAME"]
    except BaseException:
        modelname=None 
    
    # get and check parameters from config
    # ------------------------------------
    img_dir = conf["IMG_DIR"]
    assert os.path.isdir(img_dir)
    mask_dir = conf["MASK_DIR"]
    assert os.path.isdir(mask_dir)
    res_dir = conf["RESULT_DIR_1"]
    img_size = tuple(conf["IMG_SIZE"]) #(160, 160)
    print("img_sze ",img_size)
    assert img_size == (160, 160) or img_size==(512,512)
       
    num_channels=int(conf["NUM_CHANNELS"])  #3
    assert num_channels in [1,3]

    # model_data
    num_classes = int(conf["NUM_CLASSES"])   #3
    assert num_classes == 3 
    
    # use data gen for training
    # -------------------------
    use_data_gen = True #Digit&Boston standard now
    
    # training
    # --------
    epochs = int(conf["EPOCHS"]) #12
    #assert epochs == 12
    
    batch_size = int(conf["BATCH_SIZE"]) #32
    #assert batch_size ==32
    batch_size=4
   
    try:
        max_epochs=int(conf["MAX_EPOCHS"])
    except BaseException:
        max_epochs=None    
    max_epochs=None
    
    # training split
    try:
        max_samples=int(conf["MAX_SAMPLES"])
    except BaseException:
        max_samples = None    
    
    test_split_ratio = float(conf["TEST_SPLIT_RATIO"])    #0.2
    assert test_split_ratio < 1.0
    
    
    # Override parameters
    # -------------------
    modelname = "BM_DenseB_008D_169_Tensorboard"
    max_samples=None#1024*5
    batch_size = 8
    img_size=(256,256)
    epochs=12
    loss_weights = None#np.asarray([1.0,1.0,1.0], dtype=np.float32)
    epoch_evaluate = 0
    network_ID=169
    num_slices=None #DIGIT 230425 changed to none
    growth_rate = 8
    data_augment=False
    use_trans=True
    
    
    filters=None
    
    
    
    # set and select "modes" of pipeline
    # ----------------------------------
    use_init=True
    use_train=False
    use_eval=True
    
    # instantiate class with checked parameters only
    # ----------------------------------------------
    c_model = BM_Train_DenseB008(
                 img_dir=img_dir,
                 mask_dir=mask_dir,
                 res_dir = res_dir,
                 # input data
                 img_size = img_size,#(160, 160),
                 num_channels=num_channels,#3, 
                 # model_data
                 num_classes = num_classes,#3, 
                 # use data gen for training
                 # -------------------------
                 use_data_gen = True,
                 data_augment = data_augment, 
                 # architecture
                 # ------------
                 network_ID=network_ID,
                
                 # network details
                 growth_rate=growth_rate,
                 use_trans=use_trans,
                 filters=filters,
                 # training
                 # --------
                 epochs = epochs,#12,
                 batch_size = batch_size,#32,
                 # training split
                 max_samples=max_samples,
                 max_epochs=max_epochs,
                 test_split_ratio = test_split_ratio,
                 loss_weights=loss_weights,
                 # modelname
                 modelname=modelname)
    
    print("training step")
    # Pipeline for initializing 
    # -------------------------
    if use_init:
        print("Initialize model")
        # Initialze a log dir according to modelname
        c_model.init_log_dir()
        c_model.init_callbacks() 
        c_model.init_img_dir()
        c_model.init_seg_dir()
        c_model.init_pred_dir()
        c_model.init_prob_dir()
        
    if use_eval:
        # print("Predict model")
        c_model.load_model()
        # not necessary to compile model for prediction
        # ---------------------------------------------
#        c_model.compile_model()

        c_model.load_weights()
        (_,test_gen, val_gen) = c_model.get_data_gen()
                
        c_model.eval_predict_model()
        c_model.eval_save_predictions()
    print("Program terminated")
    
'''
-----------------------------------------------------
'''      
