'''
Created on 15 Jun 2021

@author: digit
'''
import numpy as np
import os
from time import gmtime, strftime
import logging

# Init logging
format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

# import tf and Keras
# -------------------
import tensorflow 
# Own imports
# -----------
# from DigitNvidia import get_gpu_usage, get_cpu_usage
#from DigitKerasPlotHistory import DigitKerasPlotHistory

#class DigitKerasCallbackHistory(DigitKerasPlotHistory, tf.keras.callbacks.Callback):
class DigitKerasCallbackHistory(tensorflow.keras.callbacks.Callback):
    
    def __init__(self, log_dir, history_file, modelname):
        
        self.log_dir = log_dir
        self.history_file = history_file
        self.modelname = modelname
        
        self.epoch_cnt=0
        self.batch_cnt=0
        self.batch_print_step=50
        self.batch_print_all=30
        
        # init base class
        # ---------------
        super().__init__()
        print("Init class: DigitKerasCallbackHistory")
        
        
    def init(self):    
        # Callback init training
        print("Initializing HistoryCallback")
        try:
            self.init_figure_history()
            self.init_data_history(history_file=self.history_file)
            self.plot_loss_acc_history(block=False)
            print("HistoryCallback Figue initialized")
        except BaseException as e:
            print("Initializing HistoryCallback error ",str(e))    
        
    
    def get_init_epoch(self) -> int:
        
        # finds the last epoch of previously training runs 
        # from the csv file in the log-dir
        # otherwise the train_loop will be responsible for that
        # -----------------------------------------------------
        if self.history_file is None:
            self.history_file= os.path.join(self.log_dir, self.modelname+'_log.csv')
        filename = self.history_file
                       
        ret = 0
        try:
            if os.path.isfile(filename):
                # Digit 01.2022
                # -->Brute force programming, please switch to pandas
                # ---------------------------------------------------
                pd = np.genfromtxt(filename,delimiter=';')
                # No data found yet
                # -----------------
                if len(pd.shape)==1:
                    ret = 0
                else:    
                    epochs = np.asarray(pd[1:,0])
                    ret = int(epochs.max())
                
        except BaseException as e:
            raise ValueError("get_init_epoch Error %s"%str(e))
        
        print("Initial epoch found as %d "% ret)
        # logging.info("Initial epoch found as %d "% ret)
        
        self.initial_epoch = ret 
        return ret 
    
    
        
    def on_train_batch_end(self, batch, logs=None):
        try:
            initial_epoch = 0#self.initial_epoch
            epochs = 0#super.epochs#0#self.get_epochs()
            try:
                lr = float(self.model.optimizer.lr.numpy())
            except BaseException as e:
                logging.warn("model.optimizer.lr.numpy error %s ",str(e))  
                lr = 0.001    
            loss = float(logs["loss"])
        except BaseException as e:
            print("On train batch_end error ",str(e))
            initial_epoch=0
            epochs=0
            lr = 0.0
            loss=0.0
        try:
            # mem = "GPU %s CPU%s"%(np.array2string(get_gpu_usage()),np.array2string(get_cpu_usage()))
            # DIGIT Jan 2023 changed to GPU only:
            mem = "GPU %s "%(np.array2string(get_gpu_usage()))," @PID %s"%(str(os.getpid()))
            
        except BaseException as e:
            print("On batch end error ",str(e))
            mem = "Error"
        try: 
            if self.batch_cnt < self.batch_print_all:
                print("init %2d epoch %2d/%2d batch %4d loss= %4.6lf with lr= %1.8lf; %s"%(initial_epoch, self.epoch_cnt, epochs, batch, loss, lr, str(mem)))
            else:
                if self.batch_cnt > 0 and np.mod(self.batch_cnt, self.batch_print_step)==0:    
                    print("init %2d epoch %2d/%2d batch %4d loss= %4.6lf with lr= %1.8lf; %s"%(initial_epoch, self.epoch_cnt, epochs, batch, loss, lr, str(mem)))
                    
        except BaseException as e:
            print("On_train_batch_end Error ",str(e))
        self.batch_cnt+=1
        
        #print("For batch {}, loss is {:7.2f}.".format(batch, logs["loss"]))
        #print("\tOn batch end memory GPU ",get_gpu_usage(), " CPU ",get_cpu_usage())
        
    def on_test_batch_end(self, batch, logs=None):
        pass
        #print("on_test_batch_end")
        #print("For batch {}, loss is {:7.2f}.".format(batch, logs["loss"]))
    
    def on_epoch_end(self, epoch, logs=None, verbose=True):
        epoch = int(epoch)
        self.epoch_cnt = epoch
        
        if verbose: print("History on epoch end ",epoch)
        if verbose: print("*"*200)
               
        # get keys
        # --------
        keys  = logs.keys()
        if verbose: print("On_epoch_end keys ",keys)
        # init values
        # -----------
        acc=0.0;loss=0.0;lr=0.0;val_acc=0.0;val_loss=0.0
        sdate=strftime("%d.%m.%Y", gmtime())
        stime=strftime("%H_%M_%S", gmtime())
        
        # extract data
        # ------------
        try:
            #
            if 'loss' in keys: loss = float(logs["loss"]) 
            if 'accuracy'      in keys: acc =       float(logs["accuracy"]) 
            if 'acc'           in keys: acc =       float(logs["acc"]) 
            
            if 'val_loss'      in keys: val_loss =  float(logs["val_loss"])
            if 'val_accuracy'  in keys: val_acc =   float(logs["val_accuracy"]) 
            if 'val_acc'       in keys: val_acc =   float(logs["val_acc"]) 
            
            if 'lr'            in keys: 
                lr = float(logs["lr"]) 
            else: 
                try:
                    lr = self.model.optimizer.lr.numpy()
                except BaseException as e:
                    logging.warn("model.optimizer.lr.numpy error %s ",str(e))    
            if not os.path.isfile(self.history_file):
                with open(self.history_file, 'a+') as f:
                    header ="epoch;acc;loss;lr;val_acc;val_loss;date;time"
                    print("History file %s initialized with %s"%(self.history_file, header))
                    f.write(header)
                    
            with open(self.history_file, 'a+') as f:
                line="%d;%lf;%lf;%lf;%lf;%lf;%s;%s\n"%(int(epoch+1),acc,loss,lr,val_acc,val_loss,sdate,stime)
                f.write(line)
                
        except BaseException as e:
            raise ValueError("History callback on_epoch_end error ",str(e))    
        
        # update chart --> evaluate thread
        # --------------------------------         
        try:
            # update chart with new values
            # ----------------------------
            # self.append_history(epoch, acc, val_acc, lr,  loss, val_loss)
            pass
        except BaseException as e:
            print("On epoch end error ", str(e))
    
    def on_train_begin(self, logs=None):
        # 
        try:
            pass
        except BaseException as e:
            raise ValueError("HistoryCallback init Training ",str(e))    
    
    def on_train_end(self, logs=None):
        try:
            pass
        except BaseException as e:
            print("History on train end error ",str(e))

'''
-------------------------------------------------------------------------------------
'''                            
