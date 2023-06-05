'''
Created on 2 Feb 2021

@author: digit
'''
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_FILE = os.path.abspath(__file__)

import numpy as np
import gc
from time import gmtime, strftime
import logging
import io
import matplotlib.pyplot as plt

# Init logging
format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")


# Confusion matrix
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import roc_auc_score as sk_roc_auc_score

# import tf and Keras
# -------------------
import tensorflow as tf #used for load_model only since "model" is not instanciated yet
from tensorflow.keras import Model
from tensorflow.keras import metrics

# Own imports
# -----------
from DigitKerasCallbacks_dkube import DigitKerasCallbacks
# from DigitKerasEval import DigitKerasEval
from DigitKerasTensorBoard import DigitKerasTensorboard

from DigitNvidia import get_gpu_usage
#from DigitFile import get_filename_only
from BM_ImagePil_dkube import save_mask, save_img

# DIGIT 230223
class DigitKerasModel(DigitKerasCallbacks, DigitKerasTensorboard):#, DigitKerasEvaluate):
#class DigitKerasModel(DigitKerasCallbacks):#, DigitKerasEvaluate):
    
    def __init__(self, dirname, modelname, log_dir=None,
                # parameters for learning step_decay
                # -----------------------------------
                # initial learning rate
                initial_lrate = 0.001,
                # multipicator after epoch_drop steps
                drop_factor = 0.85,
                # number of epochs until a reduction occurs
                epochs_constant = 20.0,
                # data for evaluation after epoch
                epoch_evaluate=0):
        
        # paths and names
        # ---------------
        self.dirname = dirname
        self.modelname = modelname
        self.log_dir = log_dir
        
        self.pred_dir = None
        self.prob_dir = None
        self.seg_dir = None
        
        self.pred_img_dir = None
        
        env = os.environ['CONDA_DEFAULT_ENV']
        print("ModelBase dirname    ", dirname)
        print("ModelBase model_name ", modelname)
        print("Model environment    ", env)
        
        if not env in ["DIGIT37", "DIGIT39"]:
            print("Mismatch environment ",env)
            
        
        # data for learning rate decay
        # ----------------------------
        self.initial_lrate = initial_lrate
        self.drop_factor = drop_factor
        self.epochs_constant = epochs_constant
        
        # data for evaluation after epoch
        self.epoch_evaluate = epoch_evaluate
        
        # init members
        # ------------
        self.initial_epoch=0
        
        
        DigitKerasCallbacks.__init__(self, dirname=self.dirname, modelname=self.modelname,log_dir=self.log_dir)
        DigitKerasTensorboard.__init__(self)
        self.writers =  self.eval_summary_init_writers(log_dir = self.log_dir)

        super().__init__(dirname=self.dirname, modelname=self.modelname,log_dir=self.log_dir)
        
        # super(ChildB, self).__init__()
                        
    
    

    def init_dirs(self):
        # init log_dir
        # ------------
        self.init_log_dir()
        self.init_pred_dir()
        self.init_prob_dir()
        self.init_seg_dir()
        self.init_img_dir()
        self.init_slices_dir()
    

    
    
    def load_weights(self, filename=None, by_name=False, verbose=1)->object:
        # load weights from best weights in log-dir 
        # or otherwise from the "local" weights
        # -----------------------------------------
        
        if filename is None:
            # check start weight
            # ------------------
            if os.path.isfile(self.startweight_file):
                filename = self.startweight_file
                if verbose: logging.info("Start weights will be retrieved from %s "%filename)
            else:
                # check best weight 
                # -----------------
                if os.path.isfile(self.bestweight_file):
                    filename = self.bestweight_file
                    if verbose: logging.info("Best weights will be retrieved from %s "%filename)
                else:    
                    # finally use local weight file created during initialisation
                    # -----------------------------------------------------------
                    filename = os.path.join(self.dirname, self.modelname + "_weights.h5")
                    if verbose: logging.info("Weights have to be retrieved from %s "%filename)
        if os.path.isfile(filename):
            print("DIGIT: load_weights: by_name is set to [%d] "%by_name)      
            logging.info("Load weights by name= %d from \n%s "%(by_name, filename))
            self.model.load_weights(filename, by_name=by_name)
            logging.info("Model weights were retrieved from \n %s "%filename)
        else:
            raise ValueError("Weight file %s does not exist"% filename)
            return None
        
        
        return self.model
    
    def load_model(self, filename=None, compile=False, verbose=True)->object:
    
        # load (training-)model from local
        # --------------------------------
        filename = os.path.join(self.dirname, self.modelname + ".h5") if filename is None else filename
        
        if os.path.isfile(filename):
            if verbose: logging.info("Model will be retrieved from %s compile = %d"%(filename,compile))
            
            # DIGIT: A model with custom loss function cannot be saved yet (01.2022)
            # Therefore compile = false during loading, compiling necessary
            # -------------------------------------------------------------
            self.model = tf.keras.models.load_model(filename, compile=compile)
            if verbose: logging.info("Model %s was retrieved from %s "%(self.modelname, filename))
            return self.model
        else:
            print("Model file %s does not exist "%filename)
            
        return self.model    
    
    
    
    def get_name(self, filepath):
        return os.path.basename(filepath).split(".")[0]
    
    
    def eval_save_predictions(self, epoch= 9, eval_gen=None, y_pred=None, y_prob=None, max_save=32, verbose=True):
        
        max_save = max_save or np.iinfo(np.uint32).max

        
        # predictions and probabilities
        y_pred = self.y_pred if y_pred is None else y_pred
        y_prob = self.y_prob if y_prob is None else y_prob
        
        #test_gen=self.test_gen if test_gen is None else test_gen
        
        # DIGIT: Which Datagen is used
        # ----------------------------
        # logging.info("Eval_save_predictions for %s"%eval_gen.name)
        
        eval_gen = self.val_gen
        # Masks
        y_test  = eval_gen.y(      maxitems=max_save) #Ony get max_save items
        y_files = eval_gen.y_files(maxitems=max_save)
        
        # Images
        # ------
        x_test  = eval_gen.x      (maxitems=max_save) #Ony get max_save items
        x_files = eval_gen.x_files(maxitems=max_save)
        
        
        # save predictions
        # ----------------
        logging.info("Saving %d predictions to %s"%(len(y_files), self.pred_dir))
        if verbose: print("y_pred.shape ",y_pred.shape)
        
        # save predictions
        # ----------------
        imgs=[]; names=[]
        for col, file in enumerate(y_files):
            
            if col >= max_save:
                break
            # get slice
            # ---------
            if self.num_slices is None:
                pred = y_pred[col,:,:]
            else:
                pred = y_pred[col,:,:,0]
            # Replace in filename: seg with pred
            # ----------------------------------
            file=str(file)
            if file.find('seg') > -1:
                file = file.replace('seg', 'pred')
                
            print(self.pred_dir)
            # self.pred_dir = "/media/BostonNFSAdele/BostonMedicalWorkbench/sahil/__BostonMedical_Source/BM_Train_2DH39_230103A/Digit_Train_B001/_architecture/BM_DenseB_008J_169_Tensorboard/logs/pred"
            print(file)
            file = os.path.join(self.pred_dir, file)
            #                        --------
            # save prediction
            # ---------------
            save_mask(pred, file)
            
            # add to list for summaries
            # -------------------------
            # (32, 256, 256, 1) 4 0 255 int64
            pred = np.asarray(pred, dtype=np.float32)
            pred = pred / 255.
            imgs.append(self.eval_check_mask(pred))
            names.append(self.get_name(file))

        # Summary predictions
        # -------------------
        logging.info("Summarize %d predictions "%len(y_files))
        self.eval_summary_pred(imgs, names, epoch)    
        #                 ----     
        
        
        # Next step: probabilities
        # ------------------------
        logging.info("Saving %d probabilities to %s"%(len(y_files), self.prob_dir))
        if verbose: print("y_prob.shape ",y_prob.shape)
        imgs=[]; names=[]
        for col, file in enumerate(y_files):
            if col >= max_save:
                break
            # get slice
            # ---------
            if self.num_slices is None:
                prob=y_prob[col,:,:]
            else:
                prob=y_prob[col,:,:,0]    
            
            # Replace in filename: seg with prob
            # ----------------------------------
            file=str(file)
            if file.find('seg') > -1:
                file = file.replace('seg', 'prob')
            # self.prob_dir = "/media/BostonNFSAdele/BostonMedicalWorkbench/sahil/__BostonMedical_Source/BM_Train_2DH39_230103A/Digit_Train_B001/_architecture/BM_DenseB_008J_169_Tensorboard/logs/prob"
            file = os.path.join(self.prob_dir, file)
            #                        --------
            
            # save prob image as colour
            # -------------------------
            try:
                save_img(prob, file, colour=True)
            except BaseException as e:
                print("Save Probs: Save_img error ",str(e))    
            
            # add to list for summaries
            # -------------------------
            # (32, 256, 256, 1) 4 0.0 1.0 float32
            # probs are [0,1]
            imgs.append(self.eval_check_image(prob))
            names.append(self.get_name(file))
            
        # Summary probabilities
        # ---------------------
        logging.info("Summarize %d probabilites "%len(y_files))
        self.eval_summary_prob(imgs, names, epoch)    
        

        # save masks
        # ----------
        logging.info("Saving %d segmentations to %s"%(max_save, self.seg_dir))
        if verbose: print("y_test.shape ",y_test.shape)
        
        imgs=[]; names=[]
        for col, file in enumerate(y_files):
            if col >= max_save:
                break
            # get slice
            # ---------
            if self.num_slices is None:
                msk=y_test[col,:,:]
            else:
                msk=y_test[col,:,:,0]    
            # self.seg_dir = "/media/BostonNFSAdele/BostonMedicalWorkbench/sahil/__BostonMedical_Source/BM_Train_2DH39_230103A/Digit_Train_B001/_architecture/BM_DenseB_008J_169_Tensorboard/logs/seg"
            file = os.path.join(self.seg_dir, file)
            #                        ---
            # save mask
            # ---------   
            save_mask(msk, file)
        
            # add to list for summaries
            # -------------------------
            # (32, 256, 256, 1) 4 0 255 int16
            msk = np.asarray(msk, dtype=np.float32)
            msk = msk/255.
            imgs.append(self.eval_check_mask(msk))
            names.append(self.get_name(file))
        
        # Summary segmentations
        # ---------------------
        #(32, 256, 256, 1) 4 0 255 int16
        
        logging.info("Summarize %d segmentations "%len(y_files))
        self.eval_summary_seg(imgs, names, epoch)    
        # self.pred_img_dir = "/media/BostonNFSAdele/BostonMedicalWorkbench/sahil/__BostonMedical_Source/BM_Train_2DH39_230103A/Digit_Train_B001/_architecture/BM_DenseB_008J_169_Tensorboard/logs/img" 
        # save_img
        # --------
        logging.info("Saving %d images to %s"%(max_save, self.pred_img_dir))
        imgs=[]; names=[]
        for col, file in enumerate(x_files):
            if col >= max_save:
                break
            # get slice
            # ---------
            
            # DIGIT: This is the original code, for what ever reason
            # ------------------------------------------------------
            img=x_test[col,:,:,0]
            #                     ###
            if self.num_slices is None: 
                img=x_test[col,:,:,0]
                #                     ###
            else:
                img=x_test[col,:,:,0]
                #                     ### ###    
            #print("save_img ",img.shape)
            file = os.path.join(self.pred_img_dir, file)
            #                        ------------   
            try:
                save_img(img, file)   
                
            except BaseException as e:
                print("img_arr ", img.shape, file)
                print("eval_save_predictions save_img error ",str(e))
                break     
            # add to list for summaries
            # -------------------------
            #(32, 256, 256, 1) 4 0.0 255.0 float32
            img = np.asarray(img, dtype=np.float32)
            img = img/255.
            imgs.append(self.eval_check_image(img))
            names.append(self.get_name(file))
        
        # Summary images
        # --------------
        logging.info("Summarize %d images "%len(y_files))
        self.eval_summary_img(imgs, names, epoch)    
        
        # clean memory
        x_test=None
        y_test=None  
        x_files=None
        y_files=None
        imgs=[]; names=[]
        
        
            
        
    def eval_predict_model(self, eval_gen=None, ind=None, verbose=True):
        
        # load weights, taken from mother class
        # -------------------------------------
        
        logging.info("Load weights skipped")
        #self.load_weights(verbose=True)
        
        if eval_gen is None:
            logging.info("eval_predict_model eval_gen set to val_gen")
            eval_gen = self.val_gen
        else:
            logging.info("eval_predict_model eval_gen taken over")
                
        if verbose: print("Predict model eval_gen ",len(eval_gen))
        try:
            #test_gen=self.test_gen#DigitDebug if test_gen is None else test_gen
            #if ind is not None:
            #    test_gen = test_gen[int(ind)]
            cnt = len(eval_gen)
            
            logging.info("Start prediction Datagen %s for %d slices"%(eval_gen.name, cnt))
            
            y_pred = self.eval_predict_step(eval_gen)#DigitDebug, Dec22 cja
            if y_pred is None:
                logging.info("eval_predict_model stopped eval_predict step returned None")
                print(       "eval_predict_model stopped eval_predict step returned None")
                
                return None    
                
        except BaseException as e:
            # DIGIT: Maybe no error raised
            # raise ValueError("predict_model error ",str(e))        
            logging.info("eval_predict_model stopped due to error %s"%str(e))
            print(       "eval_predict_model stopped due to error %s"%str(e))
            
            return None    
            
        
        logging.info("Model predicted test_gen for %d items"%(y_pred.shape[0]))
        if verbose: print("y_pred raw ",y_pred.shape, y_pred.dtype, y_pred.min(), y_pred.max())
        # Because Y_pred is an array of probabilities, we have to convert it to one vector of indices
        # This is where the predictions are converted to an integer 
        
        # DIGIT: self.num_classes taken from Model-definition class
        
        # Output for 2D
        # y_pred raw  (64, 512, 512, 3) float32 0.0 1.0
        # y_pred raw  (batch_size*len(test_gen), 512, 512, 3) float32 0.0 1.0
        # ---------------------------------------------
        

        # Planned raw output for 3D from model.predict
        # y_pred raw  (batch_size*len(test_gen), num_slices, 512, 512, num_classes) float32 0.0 1.0
        # -------------------------------------------------------------------------------

        # Transformed predictions for 3D
        # y_pred      (batch_size*len(test_gen), num_slices, 512, 512, 1) float32 0.0 1.0
        # -------------------------------------------------------------------------------
        
        # y_prob raw  (64, 512, 3) float32 1.1871151e-17 0.99999905
        
        try:
            #(32, 256, 256, 3)
            print("eval_predict_model y_pred ",y_pred.shape)
            # DIGIT changed 230502
            y_prob = y_pred
            #y_prob = y_pred[:,:,:, self.num_classes-1]
            print("y_prob raw ",y_prob.shape, y_prob.dtype, y_prob.min(), y_prob.max())
        except BaseException as e:
            raise ValueError("predict model y_prob error ",str(e))

        # This is for 2D model only
        # -------------------------
        if self.num_slices is None:
            y_pred = np.argmax(y_pred, axis = -1)
            print("eval_predict_model_2D ",y_pred.shape)
        # DIGIT: 3D model, as of Jan 2023 does the same
        # ---------------------------------------------
        else:
            # Transform raw predictions to classes-array
            # ------------------------------------------
            y_pred = np.argmax(y_pred, axis = -1)
            print("eval_predict_model_3D ",y_pred.shape)
            
            # y_pred.shape = (32=len(test_gen)/batch_size*, 256, 256, 12=num_slices)
            # y_pred      (batch_size*len(test_gen), num_slices, 512, 512, 1) uint8 0.0 1.0
            # 1 is because of sparse coding of results: One array with numbers [0,1,2]
            # -----------------------------------------------------------------------------
            
        if verbose:
            pass
            #unique, count = np.unique(y_pred,return_counts=True)        
            #print("y_pred res ",y_pred.shape, y_pred.dtype, y_pred.min(), unique, count, y_pred.max())
            
        # assign y_pred to class member
        # -----------------------------
        self.y_pred=y_pred
        # assign y_prob to class member
        # -----------------------------
        self.y_prob=y_prob
        
        return y_pred
    
    #@tf.function
    def eval_predict_step(self, val_gen=None, verbose=False):
        
        # DIGIT: Jan 2022
        # commented this because not needed, predict is called with val_gen
        # test_gen = self.test_gen if test_gen is None else test_gen
        
        # DIGIT: Nov 2022
        # This allocates len(test_gen) slices
        # 
        try:
            logging.info("Eval_predict step Datagen %s, batch_size  %d "%(val_gen.name, val_gen.batch_size))
            logging.info("Eval_predict step Datagen %s  items       %d "%(val_gen.name, len(val_gen)))
            
            y_pred= self.model.predict(val_gen, 
                                       batch_size = val_gen.batch_size,#DigitDebug changed
                                       verbose=verbose)
            if y_pred is not None:
                pass
                logging.info("Eval_predict step Datagen  y_pred.shape %s "%str(y_pred.shape))
                logging.info("Eval_predict step Datagen  y_pred.shape %s "%str(y_pred.dtype))
            else:    
                logging.info("Eval_predict step Datagen  Error y_pred is None")
                return None
                
        except BaseException as e:
            logging.info("eval_predict_step error %s"%val_gen.name)
            logging.info("%s"%str(e))
            print("eval_predict_step error ",val_gen.name, str(e))
            return None    
        return y_pred
    
        
    def eval_save_conf_matrix(self, epoch, cnf_matrix, normalize=False, verbose=True):
        epoch=int(epoch)
        if normalize:
            filename = os.path.join(self.log_dir,"%s_cnf_matrix_n.csv"%self.modelname)
        else:
            filename = os.path.join(self.log_dir,"%s_cnf_matrix.csv"%self.modelname)
                
        #self.cnf_matrix_file = filename #DIGIT 230224 not used
        
        if normalize:
            cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
        #else:
        #    cnf_matrix = cnf_matrix.astype('int')
            
        # ravel starts top left at [0,0]and goes right row by row
        # -------------------------------------------------------
        cnf_values = np.asarray(cnf_matrix.ravel())
        
        
        str_epoch="%d;"%epoch
        if normalize:
            str_values = ';'.join(['%lf' % num for num in cnf_values])
        else:
            str_values = ';'.join(['%d'  % num for num in cnf_values])
                
        line = str_epoch+str_values
        
        if verbose: logging.info("Save conf_matrix %d %s"%(epoch, line))
        # append cnf for epoch
        # -------------------
        with open(filename, 'a+') as f:
            f.write(line+"\n")     
    
 
    def eval_print_confusion_matrix(self, y_pred=None, y_test=None, normalize=False, verbose=True):
        
        # Get data
        # --------
        y_pred = self.y_pred if y_pred is None else y_pred
        y_test = self.y_test if y_test is None else y_test
        
        # Prepare data for conf
        # ---------------------
        # reload test data if not set
        # ---------------------------
        if y_test is None:
            y_test = self.val_gen.y()
            #self.y_test = y_test
            
        print("ConfMat y_test ",y_test.shape, y_test.dtype)
        print("ConfMat y_pred ",y_pred.shape, y_pred.dtype)
        assert y_test.shape == y_pred.shape, "Mismatch y_test.shape == y_pred.shape"
        
        # ravel data to 1dim arrays
        # -------------------------
        y_test = y_test.ravel()
        y_pred = y_pred.ravel()

        # create and return cnf_matrix
        # ----------------------------
        np.set_printoptions(precision=2)
        
        # Because Y_pred is an array of probabilities, we have to convert it to one hot vectors 
        #y_pred = np.argmax(y_pred, axis = 1)
        
        print("y_test, y_pred ", np.asarray(y_test).shape, np.asarray(y_pred).shape)
        assert np.asarray(y_test).shape == np.asarray(y_pred).shape, "Mismatch y_test.shape vs y_pred.shape"
        assert len(y_test.shape)==1, "Mismatch len(y_test.shape)==1"
        assert len(y_pred.shape)==1, "Mismatch len(y_pred.shape)==1"
                   
        '''
        max_conf = 50000#np.min(y_test.shape[0]-1 ,500000)#000)
        print("max_conf ",max_conf)
        
        y_test = y_test[:max_conf]
        y_pred = y_pred[:max_conf]
        '''
        if verbose:
            test_val, test_cnt = np.unique(y_test, return_counts=True)
            pred_val, pred_cnt = np.unique(y_pred, return_counts=True)
            
            print("cnf_matrix input test ",y_test.shape, y_test.min(), test_val, test_cnt, y_test.max())
            print("cnf_matrix input pred ",y_pred.shape, y_pred.min(), pred_val, pred_cnt, y_pred.max())
        
        #if unique_test != unique_pred:
        #    print("Value mismatch test/pred ",unique_test, unique_pred)
        try:
            # confmatrix via sklearn
            cnf_matrix = (sk_confusion_matrix(y_test, y_pred))
            # confmatri via tensorflow
            tf_cnf_matrix = tf.math.confusion_matrix(y_test, y_pred, dtype=tf.dtypes.int32, num_classes=self.num_classes)
            tf_cnf_matrix = tf_cnf_matrix.numpy()
            
        except BaseException as e:
            print("sk_confusion error ",str(e))    
            cnf_matrix = np.eye(3,3)
            tf_cnf_matrix = np.eye(3,3)
        
        print("SK Cnf-matrix\n",    cnf_matrix)
        print("TF Cnf-matrix\n", tf_cnf_matrix)
        if normalize:
            cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
        else:
            cnf_matrix = cnf_matrix.astype('int')
        
        #print("Cnf-matrix\n", cnf_matrix)
        
        np.set_printoptions(None)
        # DIGIT: the set cnf_matri is not normalized
        # ------------------------------------------
        self.cnf_matrix=cnf_matrix
        
        return cnf_matrix    

    def eval_get_slice(self, arr, col):
        if len(arr.shape)>3:
            return arr[col,:,:,0]
        else:
            return arr[col,:,:]
    
    def eval_check_image(self, img):
        if len(img.shape)==2:
            img = np.expand_dims(img, 0)
            img = np.expand_dims(img,-1)
        elif len(img.shape)==3:
            img = np.expand_dims(img, 0)
        
        if len(img.shape)!=4:
            print("Mismatch img.shape ==4 ", img.shape)
            assert len(img.shape)==4, "Mismach img.shape"
        return img
    
    def eval_check_mask(self, msk):
        msk = msk*127
        msk[0,0]=255
        msk = np.expand_dims(msk, 0)
        msk = np.expand_dims(msk,-1)
        if len(msk.shape)!=4:
            print("Mismatch img.shape ==4 ", msk.shape)
            assert len(msk.shape)==4, "Mismach img.shape"
        return msk 
        
    
      
'''
-----------------------------------------------------
'''    
if __name__ == '__main__':  
    print("Program terminated")
    
'''
-----------------------------------------------------
'''    
                
