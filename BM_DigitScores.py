'''
Created on 11 Jun 2021

@author: digit
'''
# Plotting the confusion matrix
import os
import numpy as np
from numpy import testing
import matplotlib.pyplot as plt
import itertools

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical as k_to_categorical

from sklearn.metrics import confusion_matrix as sk_confusion_matrix
#from sklearn.metrics import classification_report as sk_classification_report
#from tensorflow.python.ops.nn_ops import _calc_dilation2d_flops

WEIGHTS=np.asarray([0.78,0.65,8.57])
WEIGHTS=np.asarray([1.0,1.0,1.0])

SMALL_SIZE = 8
MEDIUM_SIZE = 14
BIGGER_SIZE = 22

# Own imports
# -----------
import DigitConfig

''' Config Json File '''
conf = DigitConfig.Config('UNet_Oxford')

from BM_ImagePil import load_img, load_mask

def reverse_categorical(y_cat):
    ret = np.asarray([np.argmax(y, axis=1, out=None) for y in y_cat])
    return ret

def arr_to_string(arr):
    arr = np.asarray(arr)
    converted_list = [str(element) for element in arr]
    #Convert each element into a string using a list comprehension
    joined_string = ";".join(converted_list)
    print(joined_string)    
    

class BM_DigitScores(object):
    def __init__(self, img_dir, msk_dir, res_dir,
                 num_classes=3):
        
        
        self.num_classes = num_classes
        
        self.img_dir=img_dir 
        self.msk_dir=msk_dir 
        self.res_dir=res_dir
        
        pass
    
    def de_categorize(self, y_cat, dtype=np.uint8):
        ori = np.asarray([np.argmax(y, axis=1, out=None) for y in y_cat])
        # set type
        ori = np.asarray(ori, dtype=dtype)
        print("decat ",ori.shape, ori.min(), np.unique(ori), ori.max(), ori.dtype)
        
        return ori

    def categorize(self, mask, num_classes, dtype=np.uint8):
        cat = k_to_categorical(mask, num_classes=num_classes)
        # set type
        cat = np.asarray(cat,dtype=dtype)
        print("cat   ",cat.shape, cat.min(), np.unique(cat), cat.max(), cat.dtype)
        assert cat.shape[-1]==num_classes
        
        return cat
    
    def get_data(self, filename = None):
        filename= 'Abyssinian_19' if filename is None else filename
        
        img_file = os.path.join(self.img_dir, filename+'.jpg')
        msk_file = os.path.join(self.msk_dir, filename+'.png')
        res_file = os.path.join(self.res_dir, filename+'.png')
        print("Load data ", filename)
        self.img = load_img(img_file)
        self.msk = load_mask(msk_file)
        self.res = load_mask(res_file)
        
        #ori   = np.copy(self.msk)
        #cat   = self.categorize(self.msk, num_classes=self.num_classes)
        #decat = self.de_categorize(cat)
        
        #testing.assert_array_equal(ori, decat,err_msg="Values of cat and ori are not equal", verbose=True) 
        #assert all(cat == ori)    
        
        #self.msk = self.categorize(self.msk, num_classes=self.num_classes)
        #self.res = self.categorize(self.res, num_classes=self.num_classes)
        
        print("Data loaded ",filename)
        
        #ret = self.calc_Dice(self.msk, self.res)
        
        return self.img, self.msk, self.res
        
        #np.count_nonzero(a==b)
    
       
    def get_conf_matrix(self, y_test, y_pred):
        
        assert y_test.shape == y_pred.shape
        cnf_matrix = np.zeros([self.num_classes, self.num_classes],dtype=np.int16)
        for irow in range(self.num_classes):
            for icol in range(self.num_classes):
                cnf_matrix[irow, icol] = int(np.count_nonzero(  np.logical_and(y_test==irow,y_pred==icol) ))
        print("cnf_matrix digit\n",cnf_matrix)
        
        
        self.cnf_matrix = cnf_matrix 
        
        # ravel starts top left at [0,0]and goes right row by row
        # -------------------------------------------------------
        cnf_values = np.asarray(cnf_matrix.ravel(), dtype=np.int16)
        #norm = cnf_values.sum()
        #cnf_values /= float(norm)
        epoch = 99
        str_epoch="%d;"%epoch
        str_values = ';'.join(['%d' % num for num in cnf_values])
        line = str_epoch+str_values
        # append cnf for epoch
        # -------------------
        filename="UN_ConfMatrix.txt"
        with open(filename, 'a') as f:
            f.write(line+"\n")     
        
        return cnf_matrix        
        
    def custom_loss(self, y_true, y_pred):
        
        # weighted_crossentropy
        # ---------------------
        print("loss: y_pred   ",y_pred.shape)
        print("loss: y_true   ",y_true.shape)
        
        y_pred_f = K.reshape(y_pred, (-1,3))
        y_true_f = K.reshape(y_true, (-1,))
    
        print("loss: y_pred_f ",y_pred_f.shape)
        print("loss: y_true_f ",y_true_f.shape)

        
        soft_pred_f = K.softmax(y_pred_f)
        soft_pred_f = K.log(tf.clip_by_value(soft_pred_f, 1e-10, 1.0))
    
        neg = K.equal(y_true_f, K.zeros_like(y_true_f))
        neg_calculoss = tf.gather(soft_pred_f[:,0], tf.where(neg))
    
        pos1 = K.equal(y_true_f, K.ones_like(y_true_f))
        pos1_calculoss = tf.gather(soft_pred_f[:,1], tf.where(pos1))
    
        pos2 = K.equal(y_true_f, 2*K.ones_like(y_true_f))
        pos2_calculoss = tf.gather(soft_pred_f[:,2], tf.where(pos2))
        
        # Here are the same parameters as above
        # Those weights add up to 10.0
        # ----------------------------
        loss = -K.mean(tf.concat([WEIGHTS[0]*neg_calculoss, 
                                  WEIGHTS[1]*pos1_calculoss, 
                                  WEIGHTS[2]*pos2_calculoss], axis=0))

        return loss.numpy()
    
    def custom_keras_loss(self, y_true, y_pred):
        y_pred = K.reshape(y_pred, (-1,3))
        y_true = K.reshape(y_true, (-1,))
        print("keras loss: y_pred   ",y_pred.shape)
        print("keras loss: y_true   ",y_true.shape)
        
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        #loss = loss.numpy()
        print("custom_keras_loss ",loss.shape)
        return loss.numpy()
    
    def get_conf_matrix_exp(self, msk=None, res=None):
        msk = self.msk if msk is None else msk
        res = self.res if res is None else res
        
        #msk = reverse_categorical(msk)
        #res = reverse_categorical(res)
        
        y_test = msk.ravel()
        y_pred = res.ravel()
        
        print("y_test ", y_test.shape, y_test.min(), np.unique(y_test), y_test.max())
        print("y_pred ", y_pred.shape, y_pred.min(), np.unique(y_pred), y_pred.max())
        
        assert y_test.shape == y_pred.shape
        
        print("Calc conf_matrix")
        cnf_matrix = (sk_confusion_matrix(y_test, y_pred))
        print("cnf_matrix sklearn\n",cnf_matrix)
        print("Conf_matrix finished")
        
        self.cnf_matrix = cnf_matrix
        return cnf_matrix
    
    def test_conf_matrix(self):    
        print("Plotting \n",self.cnf_matrix)
        
        y_test = self.msk#.ravel()
        y_pred = self.res#.ravel()
        
        
 
        y_test = np.asarray(y_test, dtype=np.float32)
        y_pred = np.asarray(y_pred, dtype=np.float32)
        
        # categorize predictions
        # ----------------------
        y_pred = k_to_categorical(y_pred, 3)
        
        print("y_test ", y_test.shape, y_test.min(), np.unique(y_test), y_test.max())
        print("y_pred ", y_pred.shape, y_pred.min(), np.unique(y_pred), y_pred.max())
        
        #loss_digit = self.custom_loss(y_test, y_pred)
        #loss_keras = self.custom_keras_loss(y_test, y_pred)
        #print("loss_digit loss_keras",loss_digit, loss_keras)
        
        self.plot_summary(self.img,self.msk, self.res, self.cnf_matrix)
    
    def plot_summary(self, img, msk, res, cm, 
                              
                              normalize=True, #if true all values in confusion matrix is between 0 and 1
                              title='Confusion matrix',
                              fig_width = 8.0,
                              cmap=plt.cm.Blues,
                              block=False):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        figure, ax_arr = plt.subplots(2, 2, sharex=False, squeeze =False,figsize=(fig_width, fig_width), num=title)
        
        
        # msk
        ax=ax_arr[0,0]
        if msk is not None:
            ax.imshow(msk)
        ax.set_title("Annotation")
        
        # res
        ax=ax_arr[0,1]
        if res is not None:
            ax.imshow(res)
        ax.set_title("Prediction")
        
        
        # image
        ax=ax_arr[1,0]
        if img is not None:
            ax.imshow(img)
        ax.set_title("Image")
        
        
        # conf matrix
        ax= ax_arr[1,1]
            
        img=ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.set_title("Confusion")
        #figure.colorbar(img, ax=ax)
           
        # set axes, labels, titles
        # ------------------------
        ax.set_title(title)
        ax.margins(0)
        ax.set_xlabel('Predictions \n [Non-target][Target-Organ][Tumour]' )
        ax.set_ylabel('True label  \n [Non-target][Target-Organ][Tumour]')
        ax.axis('on')
        ax.set_frame_on(True)
        
        ax.set_xticks(np.arange(cm.shape[0]+1)-.5, minor=False)
        ax.set_yticks(np.arange(cm.shape[0]+1)-.5, minor=False)
        #ax.grid(which="minor", color="b", linestyle='-', linewidth=3)
        ax.grid(which="major", color="black", linestyle='-', linewidth=1)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        # prepare cm-cell-elements
        # ------------------------
        row_sum = cm.sum(axis=1)
        col_sum = cm.sum(axis=0)
        print("cm sum axis=1", row_sum)
        print("cm sum axis=0", col_sum)
        if normalize:
            #for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])): 
            cm=self.normalize(cm)    
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        print("cm sum axis=1", row_sum)
        print("cm sum axis=0", col_sum)
        
        # round numbers in matrix
        cm = np.round(cm, decimals=2)
        
        # write cm-elements
        # -----------------
        thresh = cm.max() / 2. #switch black or white font
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     size=BIGGER_SIZE if i==j else SMALL_SIZE,
                     weight=True if i == j else False,
                     color="white" if cm[i, j] > thresh else "black")
        # show plot
        # ---------
        plt.show(block=True)    
    
    def normalize(self, cm):
        
        # prepare cm-cell-elements
        # ------------------------
        row_sum = cm.sum(axis=1)
        #col_sum = cm.sum(axis=0)
        print("cm sum axis=1", row_sum)
        #print("cm sum axis=0", col_sum)
        cm=cm.astype(np.float32)
        
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])): 
            
            if row_sum[i]==0:
                cm[i,j]=0 
                cm[i,i]=1
            else:
                cm[i, j] /= row_sum[i]    
        print("cm \n",cm)
        
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        print("cm_norm \n",cm_norm)
        
        row_sum = cm.sum(axis=1)
        print("cm sum axis=1", row_sum)
        #print("cm sum axis=0", col_sum)
        
        return cm
        
    def save_conf_matrix(self, filename, cnf_matrix, index, normalize=False, verbose=True):
       
        if normalize:
            row_sum = cnf_matrix.sum(axis=1)
            cnf_matrix= self.normalize(cnf_matrix)
        else:
            cnf_matrix = cnf_matrix.astype('int')
            row_sum = cnf_matrix.sum(axis=1)
            
        # ravel starts top left at [0,0]and goes right row by row
        # -------------------------------------------------------
        cnf_values = np.asarray(cnf_matrix.ravel())
        
        str_index="%d;"%index
        if normalize:
            str_values = ';'.join(['%lf' % num for num in cnf_values])
            
        else:
            str_values = ';'.join(['%d' % num for num in cnf_values])
        str_sum = ';'.join(['%d' % num for num in row_sum])        
        line = str_index+str_values+";"+str_sum
        
        if verbose: print("Save conf_matrix %s"%line)
        # append cnf for epoch
        # -------------------
        with open(filename, 'a+') as f:
            f.write(line+"\n")        
        
    
    def plot_conf_matrix(self, cm,
                              normalize=False, #if true all values in confusion matrix is between 0 and 1
                              title='Confusion matrix',
                              fig_width = 8.0,
                              cmap=plt.cm.Blues,
                              block=False):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        figure, ax_arr = plt.subplots(1, 1, sharex=False, squeeze =False,figsize=(fig_width, fig_width), num=title)
        
        
        # msk
        ax=ax_arr[0,0]
        
        
        # prepare cm-cell-elements
        # ------------------------
        if normalize:
            cm = self.normalize(cm)
            #cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # round numbers in matrix
        cm = np.round(cm, decimals=2)
        
        img=ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.set_title("Confusion")
        #figure.colorbar(img, ax=ax)
           
        # set axes, labels, titles
        # ------------------------
        ax.margins(0)
        ax.set_title(title)
        ax.set_xlabel('Predictions \n [Non-target][Target-Organ][Tumour]' )
        ax.set_ylabel('True label  \n [Non-target][Target-Organ][Tumour]')
        ax.axis('on')
        ax.set_frame_on(True)
        
        ax.set_xticks(np.arange(cm.shape[0]+1)-.5, minor=False)
        ax.set_yticks(np.arange(cm.shape[0]+1)-.5, minor=False)
        #ax.grid(which="minor", color="b", linestyle='-', linewidth=3)
        ax.grid(which="major", color="black", linestyle='-', linewidth=1)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        if normalize:
            print("cm \n",cm)
            print("cm   ", cm.min(), cm.max())
        
        # write cm-elements
        # -----------------
        thresh = cm.max() / 2. #switch black or white font
        
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            
            ax.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     size=BIGGER_SIZE if i==j else MEDIUM_SIZE,
                     weight=True if i == j else False,
                     color="white" if cm[i, j] > thresh else "black")
        # show plot
        # ---------
        plt.show(block=block)  
        
'''
--------------------
'''        
if __name__ == '__main__':
    data_path = conf["DATA_PATH"]
    if not os.path.isdir(data_path):
        raise ValueError("Data dir does not exist ",data_path)
    
    img_dir = conf["IMG_DIR"] 
    msk_dir = conf["MASK_DIR"]
    res_dir = conf["RESULT_DIR"]
    
  
    c_scores = BM_DigitScores(img_dir=img_dir,
                           msk_dir=msk_dir,
                           res_dir=res_dir)
    
    
    _,y_test, y_pred= c_scores.get_data()
    
    cm = c_scores.get_conf_matrix    (y_test, y_pred)
    #_ = c_scores.get_conf_matrix_exp(y_test, y_pred)
    c_scores.test_conf_matrix()
    plt.show(block=True)
    print("Program terminated")
    exit()
    
    '''   
    #y_test = np.asarray([0,0,0,0,1])#np.asarray(np.arange(10), dtype = np.float32)
    #y_pred = np.asarray([0,0,2,0,1])#np.asarray(np.arange(10), dtype = np.float32)
    
    y_test = msk.ravel()
    y_pred = res.ravel()
    
    assert y_test.shape == y_pred.shape
    
    cnf_matrix = (sk_confusion_matrix(y_test, y_pred))
    
    # Plot image, input mask, result mask and confusion matrix
    # ---------------------------------------------------------
    plot_confusion_matrix(img, msk, res, cnf_matrix,
                          normalize=False, 
                          title='Confusion matrix')
    print("Program terminated")
    exit()
    
    '''
    
    
    
    
    
    
    
    '''
    rep_str  = sk_classification_report(y_test, y_pred, target_names = None, output_dict=True)
    rep_dict = sk_classification_report(y_test, y_pred, target_names = None, output_dict=True)
        
    print("classification_report \n", rep_str)
    print("classification_dict\n", rep_dict)

     
    res= {
     '0': {'precision': 0.9301124980736631, 'recall': 0.9805052392169604, 'f1-score': 0.9546443117560995, 'support': 12311}, 
     '1': {'precision': 0.8845229681978799, 'recall': 0.8840231671139992, 'f1-score': 0.8842729970326408, 'support': 7079}, 
     '2': {'precision': 0.8087254371732469, 'recall': 0.7223832528180354, 'f1-score': 0.7631198434974908, 'support': 6210}, 
     'accuracy': 0.8912109375, 
     'macro avg':    {'precision': 0.8744536344815966, 'recall': 0.8623038863829983, 'f1-score': 0.8673457174287437, 'support': 25600}, 
     'weighted avg': {'precision': 0.8880600789259188, 'recall': 0.8912109375,       'f1-score': 0.8887253475055792, 'support': 25600}
     }

    '''
    
    '''
    The reported averages include 
    macro average (averaging the unweighted mean per label), 
    weighted average (averaging the support-weighted mean per label), 
    and sample average (only for multilabel classification). 
    
    Micro average (averaging the total true positives, 
    false negatives and false positives) is only shown for multi-label or multi-class with a subset of classes, 
    because it corresponds to accuracy otherwise and would be the same for all metrics.
    '''

    
        
    

    print("Program terminated")
'''
---------------------------------------------------------------------
'''    
    
    