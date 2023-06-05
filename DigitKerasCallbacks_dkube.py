'''
Created on 28 May 2021

@author: digit
'''
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_FILE = os.path.abspath(__file__)
import sys
import math
import numpy as np
from time import gmtime, strftime, sleep
import subprocess
import logging
#from multiprocessing import Process
# Init logging
format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=print, datefmt="%H:%M:%S")

# import tf and Keras
# -------------------
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler

# Own imports
# -----------
from DigitKerasCallbackCustom import DigitKerasCallbackCustom
from DigitKerasCallbackHistory_dkube import DigitKerasCallbackHistory
# from DigitNvidia import get_gpu_usage, get_cpu_usage



class DigitKerasCallbacks(DigitKerasCallbackHistory):
    
    def __init__(self, dirname, modelname, log_dir=None, history_file=None, bestweight_file=None):
        
        self.dirname   = dirname
        self.modelname = modelname
        
        print("Model dirname    ", dirname)
        print("Model model_name ", modelname)
    
        # log_dir for all callbacks and files
        # -----------------------------------
        self.log_dir = self.init_log_dir() if log_dir is None else log_dir
        
        # history_file for logging
        self.history_file = os.path.join(self.log_dir, self.modelname+'_log.csv') if history_file is None else history_file 
    
        # history_file for logging
        self.history_keras_file = os.path.join(self.log_dir, self.modelname+'_keras_log.csv') #if history_file is None else history_file 
        
        # bestweight_file
        self.bestweight_file = os.path.join(self.log_dir,self.modelname+'_bestweight.hdf5') if bestweight_file is None else bestweight_file
        
        # startweight_file
        self.startweight_file = os.path.join(self.log_dir,self.modelname+'_startweight.hdf5') 
        
        
        # Init base class
        # ---------------
        super().__init__(log_dir=self.log_dir, history_file=self.history_file, modelname=self.modelname)
        self.c_history = None
        
    def init_log_dir(self, logs='logs')->str:

        # creates log_dir if it does not exist
        # ------------------------------------
        log_dir = os.path.join(self.dirname, self.modelname)
        if not os.path.isdir(log_dir):
            print("Init model: Model_dir created %s "%log_dir)
            os.mkdir(log_dir)
            
        log_dir = os.path.join(log_dir, logs)
        if not os.path.isdir(log_dir):
            print("Init model: Model_log_dir created %s "%log_dir)
            os.mkdir(log_dir)
        
        print("Init model: log_dir confirmed %s "%log_dir)
        
        self.log_dir = log_dir
        #self.copy_model_file() #DIGIT 230224 why was this dumped?
        return log_dir
    
    def init_pred_dir(self, preds='pred')->str:
        pred_dir = os.path.join(self.init_log_dir(),preds)  
        if not os.path.isdir(pred_dir):
            os.mkdir(pred_dir)
        print("Init model: pred_dir confirmed %s "%pred_dir)
        
        self.pred_dir = pred_dir
        return pred_dir

    def init_prob_dir(self, probs='prob')->str:
        prob_dir = os.path.join(self.init_log_dir(),probs)  
        if not os.path.isdir(prob_dir):
            os.mkdir(prob_dir)
        print("Init model: prob_dir confirmed %s "%prob_dir)
        
        self.prob_dir = prob_dir
        return prob_dir
    
    def init_seg_dir(self, segs='seg')->str:
        seg_dir = os.path.join(self.init_log_dir(),segs)  
        if not os.path.isdir(seg_dir):
            os.mkdir(seg_dir)
        print("Init model: seg_dir confirmed %s "%seg_dir)
        
        self.seg_dir = seg_dir
        return seg_dir
    
    def init_img_dir(self, imgs='img')->str:
        img_dir = os.path.join(self.init_log_dir(),imgs)  
        if not os.path.isdir(img_dir):
            os.mkdir(img_dir)
        print("Init model: seg_dir confirmed %s "%img_dir)
        
        self.pred_img_dir = img_dir
        return img_dir

    def init_slices_dir(self, slices='slices')->str:
        slices_dir = os.path.join(self.init_log_dir(),slices)  
        if not os.path.isdir(slices_dir):
            os.mkdir(slices_dir)
        print("Init model: slices_dir confirmed %s "%slices_dir)
        
        slices_seg_dir=os.path.join(slices_dir,'seg')
        if not os.path.isdir(slices_seg_dir):
            os.mkdir(slices_seg_dir)
        
        slices_pred_dir=os.path.join(slices_dir,'pred')
        if not os.path.isdir(slices_pred_dir):
            os.mkdir(slices_pred_dir)
        
        slices_prob_dir=os.path.join(slices_dir,'prob')
        if not os.path.isdir(slices_prob_dir):
            os.mkdir(slices_prob_dir)
        
        # Directory for segmentations/annotations
        self.slices_seg_dir = slices_seg_dir
        # Directory for predictions
        self.slices_pred_dir = slices_pred_dir
        # Dir for probabilities
        self.slices_prob_dir = slices_prob_dir
        # Directory slices general    
        self.slices_dir = slices_dir
        
        return slices_dir
    
   
      
    def get_log_dir(self)->str:
        return self.log_dir
    
    # learning rate schedule
    # ----------------------
    def _step_decay(self, epoch)->float:
        # initial rate
        try:
            # Take parameters from mother-class
            # ---------------------------------
            initial_lrate = self.initial_lrate
            drop_factor = self.drop_factor
            epochs_constant = self.epochs_constant
        
        except BaseException as e: 
            raise ValueError("_Init step decay error  ", str(e))   
            initial_lrate = 0.1
            # multipicator after epoch_drop steps
            drop_factor = 0.5
            # number of epochs unti a reduction occurs
            epochs_constant = 10.0    
                        
        # calculate learning rate
        lrate = initial_lrate * math.pow(drop_factor, math.floor((1+epoch)/epochs_constant))
        #print("Step decay learning rate for epoch %d %2.4lf "%(epoch,lrate))
        return lrate
    
    def _on_epoch_end(self, epoch)->float:
        print("_on_epoch_end will start")
        self.on_epoch_end(epoch)
        
        print("_on_epoch_end return lr")
        try:
            lr= self.model.optimizer.lr.numpy()
            print("On_epoch_end Learning rate ", lr)
            return lr
        except BaseException as e:
            print("On_epoch_end Learning rate Error ",str(e))
            return 0.001
        
    def on_epoch_end(self, epoch)->float:
        '''
        @Digit: Brute force helper to have on_epoch_end event
        since Keras.callbacks baseclass prevents inheritence somehow
        @Digit: Open issue 01.2022
        --------------------------
        '''
        print("on_epoch_end_function ",epoch)
        epoch = int(epoch)
        # save/copy best weight file for epoch
        # ------------------------------------
        self.copy_file(epoch, file_in=self.bestweight_file)
        try:
            print("On epoch end memory GPU ",get_gpu_usage(), " CPU ",get_cpu_usage())
        except BaseException as e:    
            print("Memory usage error ",str(e))
        #if False:
        try:
            print("epoch_evaluate ",self.epoch_evaluate)
        except BaseException:
            self.epoch_evaluate = 999999    
        
        if epoch < self.epoch_evaluate:
            print("Callback: On_epoch_end: Current epoch %d skipped; evaluated at epoch %d "%(epoch, self.epoch_evaluate))
        else:     
            print("Callback: On_epoch_end called %d "%epoch)
            # Predict model
            # -------------
            try:
                # Digit: load_weights maybe not needed??
                # --------------------------------------
                #print("On epoch End Load weights")
                print("On epoch End Load no weights loaded")
                
                #self.load_weights()#DigitDebug
                
                #print("On epoch End no weights loaded")
                #print("*"*180)
                #print("On epoch End Predict model")
                
                print("DIGIT predict model in on_epoch_end changed to val_gen")
                self.eval_predict_model(eval_gen=self.val_gen, ind=None, verbose=True)
                print("DIGIT predict model in on_epoch_end predicted")
                
            except BaseException as e:
                raise ValueError("On_epoch_end on error predict_model ",str(e))
                return    

            # reload test data if not set
            # DIGIT checked 230224, because not loaded
            # ----------------------------------------
            if self.y_test is None:
                print("load y_test data from val_gen")
                self.y_test = self.val_gen.y()
            
            # DIGIT: except filenames everything is set
            # y_test, y_pred, y_prob
            # ----------------------
            
            
            # print confusion matrix
            # ----------------------
            try:
                print("On_epoch_end on train end: Eval_print_confusion_matrix")
                self.cnf_matrix = self.eval_print_confusion_matrix() #epoch end
                pass
            except BaseException as e:
                print("On_epoch_end error eval_print_confusion_matrix ",str(e))    
                    
            # save conf_matrix
            # ----------------
            try:
                self.eval_save_conf_matrix(epoch, self.cnf_matrix)
                pass
            except BaseException as e:
                print("On_epoch_end error eval_save_conf_matrix ",str(e))    

            
            # summary confusion matrix
            # ------------------------
            print("On_epoch_end : Eval_summary confusion_matrix_original") #epoch end
            self.eval_summary_conf(self.cnf_matrix, epoch, normalize=False) 
            
            print("On_epoch_end : Eval_summary confusion_matrix_normalized") #epoch end
            self.eval_summary_conf(self.cnf_matrix, epoch, normalize=True) 
            
            # summary ROC and PRC curves
            # --------------------------
            #print("On_epoch_end : Eval_summary ROC and PRC curves") #epoch end
            self.eval_summary_curves(self.y_test, self.y_pred, self.y_prob, epoch,  verbose=True)
            
            
            # save predictions
            # ----------------
            print("Digit eval_save_predictions changed from test_gen to val_gen")
            
            # Data from val_gen reloaded because filenames are needed
            # -------------------------------------------------------
            self.eval_save_predictions(epoch, eval_gen=self.val_gen)#epoch end
            
            # DIGIT: new 230301 summarize pointcloud
            # --------------------------------------
            # self.eval_summary_pcl(epoch)
            
            # DIGIT: new 230224: flush writers
            # --------------------------------
            self.eval_summary_flush_writers()
              
            
            # DIGIT: new 230301 write profiling
            # ---------------------------------
            # https://www.tensorflow.org/api_docs/python/tf/summary/trace_on      
            #tf.summary.trace_export()     
                
        '''
        @Digit
        Since this is based on LearningRate callback
        the "original" learning rate is returned
        ----------------------------------------
        '''
        try:
            lr= self.model.optimizer.lr.numpy()
            print("On_epoch_end Learning rate ", lr)
            return lr
        except BaseException as e:
            print("On_epoch_end Learning rate Error ",str(e))
            return 0.001
    
    
    def init_callbacks(self, cb_LearningRateScheduler=True,
                             cb_OnEpochEnd=True,
                             cb_EarlyStopping=False,
                             cb_TensorBoard=True,
                             cb_History=True,
                             cb_CSVLogger=True,
                             cb_Bestweight=True,
                             cb_CallbackCustom=False,
                             cb_CallbackCustomHistory=True)->object:        
        
        # initialize callbacks according to selection
        # -------------------------------------------
        callbacks=[]
        
        # learning schedule
        # -----------------
        if cb_LearningRateScheduler:
            print("Callback init LearningRateScheduler Callback initialized")
            lrate = LearningRateScheduler(self._step_decay)
            callbacks.append(lrate)
            
        # Calculate conf_matrix at end of epoch
        # @Digit: 01.2022, quite a bit of brute force, but works
        # ------------------------------------------------------
        if cb_OnEpochEnd:
            print("Callback init LearningRateScheduler for cb_OnEpochEnd Callback initialized")
            on_epoch_end = LearningRateScheduler(self._on_epoch_end) 
            callbacks.append(on_epoch_end)
        
        # early stopping
        # --------------
        if cb_EarlyStopping: #preferably False
            early_stopping = tf.keras.callbacks.EarlyStopping(patience=2)
            callbacks.append(early_stopping)
        
        # tensorboard
        # -----------
        if cb_TensorBoard: #preferably True: DIGIT230222 because DKUbe
            print("Callback init Tensorboard")
            '''
            https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/TensorBoard
            tf.keras.callbacks.TensorBoard(
                    log_dir='logs',
                    histogram_freq=0,
                    write_graph=True,
                    write_images=False,
                    write_steps_per_second=False,
                    update_freq='epoch',
                    profile_batch=0,
                    profile_batch     Profile the batch(es) to sample compute characteristics. 
                    profile_batch must be a non-negative integer or a tuple of integers. 
                    A pair of positive integers signify a range of batches to profile. 
                    By default, profiling is disabled. 
                    
                    
                    embeddings_freq=0,
                    embeddings_metadata=None,
                    **kwargs)
                
            '''
            tf_log_dir = os.path.join(self.log_dir, "_tf_log")
            tensorboard = tf.keras.callbacks.TensorBoard(
                                                    #log_dir: the path of the directory where to save the log files to be
                                                    #parsed by TensorBoard.)
                                                    log_dir = tf_log_dir,
                                                    histogram_freq=1, 
                                                    #profile_batch=[10,20], #DIGIT: New 230224
                                                    #: whether to visualize the graph in TensorBoard. The log file
                                                    #can become quite large when write_graph is set to True.        
                                                    write_graph = False)
            
            # Intialize filewriters
            # ---------------------
            # DIGIT: Here I create one file_writer per topic
            # could also be one single one. Let's see
            # ---------------------------------------
            
            '''
            self.writer = tf.summary.create_file_writer(
                            logdir=self._log_dir,
                            max_queue=None,
                            flush_millis=None,
                            filename_suffix=None,
                            name=None,
                            experimental_trackable=False
                        )

            '''

            
            
            
            if os.path.isdir(self.log_dir):
                print("Callback init tensorboard with log_dir \n%s"%self.log_dir)
                # used only if log_dir exists
                # ---------------------------
                callbacks.append(tensorboard)
                cmd = "tensorboard --logdir %s/"%self.log_dir
                
                print("Tensorboard command \n ",cmd)
                filename=os.path.join(self.log_dir, "_%s_tensorboard.cmd"%self.modelname)
                with open(filename, 'w') as f:
                    f.write(cmd+"\n") 
                    print("Tensorboard command written to ",filename)
                
                filename=os.path.join(BASE_DIR, "_%s_tensorboard.cmd"%self.modelname)
                with open(filename, 'w') as f:
                    f.write(cmd+"\n") 
                    print("Tensorboard command written to ",filename)
                
                
                print("Tensorboard cmd")
                print(cmd)
                print("-"*100)    
            else:
                raise ValueError("Error creating Tensorboard")
                    
        '''
        @Digit: Jan 2022: History and CSV_Logger are now separate files due to Keras-bug(s)
        -----------------------------------------------------------------------------------
        '''
            
        # History
        # -------
        if cb_History: #preferably True
            print("Callback init History")
            history = tf.keras.callbacks.History() #History has not arguments, raises event: on_epoch_end
            callbacks.append(history)
        
        # CSV_logger
        # ----------
        if cb_CSVLogger: #preferably True, very important
            print("Callback init CSV-Logger with history_file \n%s"%self.history_file)
            
            csvlogger = tf.keras.callbacks.CSVLogger(
                                                    #filename: filename of the csv file, e.g. 'run/log.csv'.
                                                    # @Digit: Not the "history_file
                                                    filename = self.history_keras_file,
                                                    #separator: string used to separate elements in the csv file.
                                                    separator=";",
                                                    #: append if file exists 
                                                    append= True) 
            callbacks.append(csvlogger)
        
        # bestweight
        # ----------
        if cb_Bestweight: #preferably True, very important
            print("Callback init bestweight with bestweight_file \n%s"%self.bestweight_file)
            bestweight = tf.keras.callbacks.ModelCheckpoint(filepath=self.bestweight_file, 
                                                            # Digit: Maybe switch here to val_acc
                                                            # -----------------------------------
                                                            # DIGIT: 'val_accuracy' does not work at all
                                                            # ------------------------------------------
                                                            #monitor='val_accuracy',
                                                            monitor="loss",
                                                            verbose=1,
                                                            save_best_only=True,
                                                            save_weights_only=True,
                                                            
                                                            # either use auto or take care with what you monitor
                                                            # mode='max',
                                                            mode="auto",
                                                            # save_freq: `'epoch'` or integer. When using `'epoch'`, the callback saves
                                                            # the model after each epoch. When using integer, the callback saves the
                                                            # model at end of a batch at which this many samples have been seen since
                                                            # last saving. Note that if the saving isn't aligned to epochs, the
                                                            # monitored metric may potentially be less reliable
                                                            save_freq="epoch")
            callbacks.append(bestweight)
        
        # Init Custom class
        # -----------------
        if cb_CallbackCustom: #preferably False
            print("Callback init Custom")
            custom = DigitKerasCallbackCustom(self.dirname, self.modelname, self.log_dir)
            callbacks.append(custom)
            
        # Init History plot
        # -----------------
        if cb_CallbackCustomHistory:
            print("Callback init History")
            custom_plot = DigitKerasCallbackHistory(log_dir=self.log_dir, history_file=self.history_file, modelname=self.modelname)
            
            # set history 
            # -----------
            # DigitBoston: Not included yet 26.01.2022
            # ----------------------------------------
            
            self.c_history = None#DigitKerasPlotHistory(history_file=self.history_file, modelname=self.modelname)
            
            callbacks.append(custom_plot)
        
        # set and return
        # --------------
        self.callbacks = callbacks
        
        print("Summary of callbacks")
        for cb in callbacks:
            print("", cb)
        
        return callbacks        

    def xcopy_model_file(self, filename=None, verbose=True):
        import sys
        try:
            sdate=strftime("%Y_%m_%d", gmtime())
            stime=strftime("%H_%M_%S", gmtime())
            
            main_path = os.path.abspath(str(sys.modules[self.get_model].__file__))
            main_base = os.path.dirname(main_path)
            main_file = os.path.basename(main_path)
            main_name = os.path.splitext(main_file)[0]
            main_end  = os.path.splitext(main_file)[1]
            # insert index(epoch)
            file_out = main_name + sdate+"_"+stime+main_end
            file_out = os.path.join(self.log_dir, file_out)
            # check
            
            print("Main File copied to log %s"%file_out)
            if verbose:
                print("main_path ",main_path )
                print("main_base ",main_base )
                print("main_file ",main_file )
                print("sdate     ",sdate)
                print("stime     ",stime)
                print("")
            
            # copy file
            import shutil
            shutil.copyfile(main_path, file_out)
        except BaseException as e:
            print("copy_main_file error ",str(e))


'''
-----------------------------------------------------
''' 
if __name__ == '__main__':  
    dirname = BASE_DIR
    modelname = "Test_cb"
    c_callback = DigitKerasCallbacks(dirname=dirname, modelname=modelname)
    cb_list= c_callback.init_callbacks()
    for idx, cb in enumerate(cb_list):
        print("cb ", idx, cb)
        
    writers=c_callback.init_writers(log_dir=dirname)
    print("writers ",len(writers))
    for idx, writer in enumerate(writers):
        print("writer %d %s"%(idx, str(writer)))    

    print("Program terminated")    
'''
-----------------------------------------------------
''' 

