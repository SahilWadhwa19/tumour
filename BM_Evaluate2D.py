'''
Created on 11 Jun 2021

@author: digit
'''
# Plotting the confusion matrix
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_FILE = os.path.abspath(__file__) 
import numpy as np
import matplotlib.pyplot as plt
import itertools
import time

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import metrics as k_metrics
from tensorflow.keras.utils import to_categorical as k_to_categorical

from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import classification_report as sk_classification_report
from sklearn.metrics import roc_auc_score as sk_roc_auc_score
from sklearn.metrics import recall_score as sk_recall_score
from sklearn.metrics import jaccard_score as sk_jaccard_score
from sklearn.metrics import f1_score as sk_f1_score
    
from sklearn.metrics import auc as sk_auc
from sklearn.metrics import precision_recall_curve as sk_precision_recall_curve

from BM_StartProgram import BM_StartProgram
BM_StartProgram(BASE_FILE)


# Own imports
# -----------
import DigitConfig

''' Config Json File '''
conf = DigitConfig.Config('_BostonMedical')

# from DigitNvidia import set_gpu#CPU/GPU
# set_gpu(False)
#from DigitProcess import set_max_priority
#set_max_priority()


from BM_ImagePil_dkube import load_img, load_mask
from DigitFileDir import DigitFileDir
from BM_DigitRocCurve import BM_DigitRocCurve
from BM_Digit_CUDA_ConfMatrix import cuda_conf_matrix

class BM_Evaluate2D(object):
    def __init__(self):
        pass
            
    def init_dirs(self, img_dir, msk_dir, res_dir, num_classes=3):
        
        self.img_dir=img_dir 
        self.msk_dir=msk_dir 
        self.res_dir=res_dir
        self.num_classes = num_classes
        
        pass
    
    def get_data(self, filename = None):
        filename= 'Abyssinian_17' if filename is None else filename
        
        filename = os.path.basename(filename)
        filename = os.path.splitext(filename)[0]
        
        img_file = os.path.join(self.img_dir, filename+'.jpg')
        msk_file = os.path.join(self.msk_dir, filename+'.png')
        res_file = os.path.join(self.res_dir, filename+'.png')
        
        
        print("Load data ", filename)
        self.img = load_img(img_file)
        self.msk = load_mask(msk_file)
        self.res = load_mask(res_file)
        print("Data loaded ",filename)
        return self.img, self.msk, self.res
    
    def get_metrics(self):
        my_metrics = [
            k_metrics.TruePositives(name='tp'),
            k_metrics.FalsePositives(name='fp'),
            k_metrics.TrueNegatives(name='tn'),
            k_metrics.FalseNegatives(name='fn'), 
            k_metrics.Precision(name='precision'),
            k_metrics.Recall(name='recall'),
            k_metrics.AUC(name='roc_auc', curve='ROC'),
            k_metrics.AUC(name='prc_auc', curve='PR')]#, # precision-recall curve
            #metrics.SparseCategoricalAccuracy (name='acc', dtype=None),
            #metrics.SparseCategoricalCrossentropy(name="sCCE")]
        return my_metrics
    
        
    def cpu_conf_matrix(self, y_test, y_pred, num_classes=3):
        # DIGIT: base code to be transerred into CUDA
        # -------------------------------------------
        cnf = np.zeros((num_classes, num_classes),dtype=np.uint16)
        
        for icol in range(y_test.shape[0]):
            for irow in range(y_test.shape[1]):
                idx_col = y_test[icol][irow]
                idx_row = y_pred[icol][irow]
                cnf[idx_col][idx_row]+=1
                    
        return cnf            
    
    def calc_metrics(self, msk=None, res=None):
        msk = self.msk if msk is None else msk
        res = self.res if res is None else res
        
        metrics = self.get_metrics()
        for idx, metric in enumerate(metrics):
            print("Metrics ",idx, metric.name, metric) 
        
        # Digit: Obviously normalization to [0,1] is necessary??
        # ------------------------------------------------------    
        y_test = msk.ravel()
        y_pred = res.ravel()
        
        y_test_cat = k_to_categorical(y_test, self.num_classes)
        y_pred_cat = k_to_categorical(y_pred, self.num_classes)
        
        print("y_test ",y_test_cat.shape)
        print("y_pred ",y_pred_cat.shape)

        print("y_pred", y_pred.min(), y_pred.max())
        for idx, metric in enumerate(metrics):
            try:
                res = metric(y_test_cat, y_pred_cat)
                print("res ",idx, metric.name, res.numpy(), res)
            except BaseException as e:
                print("metric error ",idx, metric.name, str(e))    
               
    def calc_report(self, msk=None, res=None):
        msk = self.msk if msk is None else msk
        res = self.res if res is None else res 
    
        # Digit: Obviously normalization to [0,1] is necessary??
        # ------------------------------------------------------    
        y_test = msk.ravel()
        y_pred = res.ravel()
    
        rep_str  = sk_classification_report(y_test, y_pred, target_names = None, output_dict=False)
        rep_dict = sk_classification_report(y_test, y_pred, target_names = None, output_dict=True)
            
        print("classification_report \n", rep_str)
        print("classification_dict\n", rep_dict)

        '''     
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
        
        Micro average (averaging the total true positives, false negatives and false positives) 
        is only shown for multi-label or multi-class with a subset of classes, 
        because it corresponds to accuracy otherwise and would be the same for all metrics.
        '''
        return rep_dict, rep_str
    
    
    def get_conf_matrix(self, msk=None, res=None, normalize=False, verbose=False):
        msk = self.msk if msk is None else msk
        res = self.res if res is None else res
        
        # conf from CPU
        # -------------
        time_start = time.perf_counter()
        cnf_matrix = self.cpu_conf_matrix(msk, res)
        time_end = time.perf_counter()
        #print("Process time digit %2.9lf [sec]"%float(time_end-time_start))
        if verbose: print("cm \n",cnf_matrix)
        
        y_test = msk.ravel()
        y_pred = res.ravel()
        
        print("y_test ", y_test.shape, y_test.min(), np.unique(y_test), y_test.max())
        print("y_pred ", y_pred.shape, y_pred.min(), np.unique(y_pred), y_pred.max())
        assert y_test.shape == y_pred.shape
        
        # conf from sklearn
        # -----------------
        #time_start = time.perf_counter()
        sk_cnf_matrix = (sk_confusion_matrix(y_test, y_pred))
        #time_end = time.perf_counter()
        #print("Process time sklean %2.9lf [sec]"%float(time_end-time_start))
        
        # conf from tensorflow
        # --------------------
        #time_start = time.perf_counter()
        tf_cnf_matrix = tf.math.confusion_matrix(y_test, y_pred, dtype=tf.dtypes.int32, num_classes=self.num_classes)
        tf_cnf_matrix = tf_cnf_matrix.numpy()
        #time_end = time.perf_counter()
        #print("Process time tensor %2.9lf [sec]"%float(time_end-time_start))
        
        # check conf-matices
        # ------------------
        assert np.allclose(sk_cnf_matrix, tf_cnf_matrix), "Mismatch sk vs. tf"
        assert np.allclose(sk_cnf_matrix,    cnf_matrix), "Mismatch sk vs cuda"
        
        # calc helping arrays
        # -------------------
        if normalize:
            cnf_matrix = np.asarray(cnf_matrix, dtype=np.float32)
            cnf_matrix = cnf_matrix / cnf_matrix.sum(axis=1)[:, np.newaxis]
        
        if verbose: print("cm \n",cnf_matrix)
        return cnf_matrix
    
    def calc_scores(self, cnf_matrix, msk=None, res=None, num_classes=3, normalize=True):
        msk = self.msk if msk is None else msk
        res = self.res if res is None else res
        
        y_test = msk.ravel()
        y_pred = res.ravel()
        
        # convert cnf_matrix to float for calculation
        # -------------------------------------------
        cnf_matrix = np.asarray(cnf_matrix, dtype=np.float32)
        if normalize:
            cnf_matrix = cnf_matrix / cnf_matrix.sum(axis=1)[:, np.newaxis]
        
        # calc helping arrays
        # -------------------
        sum_col = np.sum(cnf_matrix, axis=0)
        sum_row = np.sum(cnf_matrix, axis=1)
        sum_all = np.sum(cnf_matrix)
        
        print("cnf_matrix \n",cnf_matrix)
        print("sum_col ",sum_col)
        print("sum_row ",sum_row)
        print("sum_all ",sum_all)
        
        
        w_row   = sum_col/sum_all
        diag = np.diag(cnf_matrix)
        trace = np.sum(np.diag(cnf_matrix))
        fn = np.subtract(sum_row, diag)
        fp = np.subtract(sum_col, diag)
        tn = np.sum(diag[:-1])
        print("diag[:-1] ", diag[:-1], tn )
        
        print("sum col  ",sum_col, sum_col.dtype)
        print("sum row  ",sum_row, sum_row.dtype)
        print("diag     ",diag, diag.dtype)
        print("trace    ",trace)
        print("sum_all  ",sum_all)
        print("w_row    ",w_row)
        print("")
        print("fn (row) ",fn)
        print("fp (col) ",fp)
        print("tp (diag ",diag)
        
        # allocate arrays
        # ---------------
        prec = np.zeros(num_classes)
        recall=np.zeros(num_classes)
        dice = np.zeros(num_classes)
        tpr  = np.zeros(num_classes)
        ppr  = np.zeros(num_classes)
        fpr  = np.zeros(num_classes)

        sk_prec  =  np.zeros(num_classes)
        sk_recall = np.zeros(num_classes)
        sk_dice  =  np.zeros(num_classes)
        
        print("Prec  : diag over col_sum")
        print("Recall: diag over row_sum")
        print("Dice  : mean (prec,rec)")
        
        # sensitivity, recall, hit rate, or true positive rate (TPR)
        #TPR = recall = TP / P = TP /(TP+FN)= diag / sum_row
        #T P R = T P P = T P T P + F N = 1 − F N R {\displaystyle \mathrm {TPR} ={\frac {\mathrm {TP} }{\mathrm {P} }}={\frac {\mathrm {TP} }{\mathrm {TP} +\mathrm {FN} }}=1-\mathrm {FNR} } {\displaystyle \mathrm {TPR} ={\frac {\mathrm {TP} }{\mathrm {P} }}={\frac {\mathrm {TP} }{\mathrm {TP} +\mathrm {FN} }}=1-\mathrm {FNR} }
        
        # precision or positive predictive value (PPV)
        #PPV = prec = TP / (TP + FP) = diag / sum_col
        #P P V = T P T P + F P = 1 − F D R {\displaystyle \mathrm {PPV} ={\frac {\mathrm {TP} }{\mathrm {TP} +\mathrm {FP} }}=1-\mathrm {FDR} } {\displaystyle \mathrm {PPV} ={\frac {\mathrm {TP} }{\mathrm {TP} +\mathrm {FP} }}=1-\mathrm {FDR} }

        
        # specificity, selectivity or true negative rate (TNR)
        #T N R = T N N = T N T N + F P = 1 − F P R {\displaystyle \mathrm {TNR} ={\frac {\mathrm {TN} }{\mathrm {N} }}={\frac {\mathrm {TN} }{\mathrm {TN} +\mathrm {FP} }}=1-\mathrm {FPR} } {\displaystyle \mathrm {TNR} ={\frac {\mathrm {TN} }{\mathrm {N} }}={\frac {\mathrm {TN} }{\mathrm {TN} +\mathrm {FP} }}=1-\mathrm {FPR} }
        #TNR = specivity = TN / N = TN / (TN + FP) = 1 - FDR with FDR = FP / (FP+ TP)
        
        #negative predictive value (NPV)
        
        # Calculate own scores
        # --------------------
        for idx in range(num_classes):
            recall[idx] = diag[idx]/sum_row[idx]
            prec[idx]   = diag[idx]/sum_col[idx]
            dice[idx]   = np.mean([prec[idx], recall[idx]]) 
            tpr[idx]    = diag[idx]/sum_row[idx]
            ppr[idx]    = diag[idx]/sum_col[idx]
            fpr[idx]    = fp[idx]/(fp[idx]+diag[idx]) 
            #tnr[idx]    =  
        
        # load class_rep
        # --------------
        rep,_= self.calc_report(msk, res)
        
        
        for idx in range(num_classes):
            sk_prec  [idx] = float(rep['%d'%idx]['precision'])
            sk_recall[idx] = float(rep['%d'%idx]['recall'])
            sk_dice[idx]   = float(rep['%d'%idx]["f1-score"])
        '''    
        a=y_test
        b=y_pred
        print("a == b    ",np.count_nonzero(a == b))
        print("a != b    ",np.count_nonzero(a != b))
        print("a >  b    ",np.count_nonzero(a >  b))
        print("a <  b    ",np.count_nonzero(a <  b))
        '''
        
        print("Tumour")
        print("")
        
        print("prec    ",prec)
        print("sk_prec ",sk_prec )
        print("_"*80)
        print("rec     ",recall)
        print("sk_rec  ",sk_recall)
        print("_"*80)
        
        print("dice    ",dice)
        print("sk_dice ",sk_dice)
        print("_"*80)
        
        #Please choose another average setting, one of [None, 'micro', 'macro', 'weighted'].
        sk_jacc = sk_jaccard_score(y_test, y_pred, average= 'weighted')
        sk_f1   = sk_f1_score(     y_test, y_pred, average= 'weighted')
        print("sk_jacc    ",sk_jacc)
        print("sk_f1      ",sk_f1)
        
        print("")
        print("tpr sensitivity recall - true positive rate    **row**        ",tpr)
        print("ppr precision          - pos pred rate         **col**        ",ppr)
        print("fpr fall-out or false positive rate                           ",fpr)
        print("tnr specificity, selectivity or true negative rate            ",1.0-fpr)
        
        '''
        assert np.allclose(prec, sk_prec),"Mismatch prec"
        assert np.allclose(recall, sk_recall),"Mismatch recall"
        assert np.allclose(dice, sk_dice,rtol=1.e-1),"Mismatch sk_dice"
        '''
        print("Scores checked")
        print("*"*80)
        
         
        sk_acc = float(rep['accuracy'])
        print("sk_acc    ",sk_acc)
        #'macro avg': {'precision': 0.8744536344815966, 'recall': 0.8623038863829983, 'f1-score': 0.8673457174287437, 'support': 25600}

        sk_macro_prec   = float(rep['macro avg']['precision'])
        sk_macro_recall = float(rep['macro avg']['recall'])
        sk_macro_dice   = float(rep['macro avg']['f1-score'])
        
        'weighted avg'
        sk_weight_prec   = float(rep['weighted avg']['precision'])
        sk_weight_recall = float(rep['weighted avg']['recall'])
        sk_weight_dice   = float(rep['weighted avg']['f1-score'])

        
        
        y_test_cat = k_to_categorical(y_test, self.num_classes)
        y_pred_cat = k_to_categorical(y_pred, self.num_classes)

        m_precision=k_metrics.Precision(name='precision')
        tf_prec = m_precision(y_test_cat, y_pred_cat).numpy()
        print("tf_prec    ",tf_prec)
        
        m_recall= k_metrics.Recall(name='recall')
        tf_rec = m_recall(y_test_cat, y_pred_cat).numpy()
        print("tf_rec     ",tf_rec)

        print("Macro means average")
        print("sk_macro_prec    ",sk_macro_prec)
        macro_prec = np.mean(prec)
        print("macro_prec       ",macro_prec)
        #assert np.allclose(sk_macro_prec, macro_prec),"Mismatch macro_prec %f %f"%(sk_macro_prec, macro_prec)
        
        print("sk_macro_recall  ",sk_macro_recall)
        macro_recall = np.mean(recall)
        print("macro_recall     ",macro_recall)
        assert np.allclose(sk_macro_recall, macro_recall),"Mismatch macro_recall"
        
        
        print("sk_macro_dice    ",sk_macro_dice)
        macro_dice = np.mean(dice)
        print("macro_dice       ",macro_dice)
        assert np.allclose(sk_macro_dice, macro_dice, rtol=1.e-2),"Mismatch macro_dice"
        
        print("")
        print("Weighted averages:")
        print("sk_weight_prec    ",sk_weight_prec)
        weight_prec = np.sum(prec*w_row)
        print("weight_prec       ",weight_prec)
        
        print("")
        print("sk_weight_recall  ",sk_weight_recall)
        weight_recall = np.sum(recall*w_row)
        print("weight_recall     ",weight_recall)
        
        print("")
        print("sk_weight_dice    ",sk_weight_dice)
        weight_dice = np.sum(dice*w_row)
        print("weight_dice       ",weight_dice)
        
        #multi_class must be in ('ovo', 'ovr')
        # ovo: 0.9033047841989976
        # ovr: 0.9033047841989976
        
    def calc_auc(self, msk=None, res=None, num_classes=3):
        msk = self.msk if msk is None else msk
        res = self.res if res is None else res
        
        y_test = msk.ravel()
        y_pred = res.ravel()
        
        y_test_cat = k_to_categorical(y_test, num_classes)
        y_pred_cat = k_to_categorical(y_pred, num_classes)
         
        print("*"*80)
        print("")
        print("ROC_AUC and PRC_AUC")
        
        print("ROC-Scores")
        multi_classes=('ovo', 'ovr')
        for multi_class in multi_classes:
            sk_roc_auc = sk_roc_auc_score(y_test_cat, y_pred_cat, multi_class=multi_class)
            print("sk_roc_auc cat           ",multi_class, sk_roc_auc)
        
        
        # sk_roc_auc does not work on sparse data
        # ---------------------------------------
        
        multi_classes=('ovr', 'ovo')
        averages = [None, 'micro', 'macro', 'weighted', 'samples']
        for average in averages:
            for multi_class in multi_classes:
                sk_roc_auc = sk_roc_auc_score(y_test_cat, y_pred_cat, average=average, multi_class=multi_class)
                print("sk_roc_auc               ",multi_class, average, sk_roc_auc)
        
        
        m_roc_auc = k_metrics.AUC(name='roc_auc', curve='ROC')
        tf_roc_auc = m_roc_auc(y_test_cat, y_pred_cat).numpy()
        print("tf_roc_auc            ",tf_roc_auc)
        
        print("")
        print("-"*80)
        print("Recall scores")
        #Please choose another average setting, one of 
        averages = [None, 'micro', 'macro', 'weighted', 'samples']
        for average in averages:
            sk_prc_auc = sk_recall_score(y_test_cat, y_pred_cat, average=average)
            print("sk_prc_auc cat %s \t %s           "%(str(average), str(sk_prc_auc)))
        
        
        #Please choose another average setting, one of 
        averages = [None, 'micro', 'macro', 'weighted']
        for average in averages:
            sk_prc_auc = sk_recall_score(y_test, y_pred, average=average)
            print("sk_prc_auc %s \t %s           "%(str(average), str(sk_prc_auc)))
        
        
        
        m_prc_auc=k_metrics.AUC(name='prc_auc', curve='PR')
        tf_prc_auc = m_prc_auc(y_test_cat, y_pred_cat).numpy()
        print("tf_prc_auc            ",tf_prc_auc)
        
        # multiclass format is not supported
        #precision, recall, _ = sk_precision_recall_curve(y_test_sort, y_pred_sort)
        #sk_precrec_score = sk_auc(recall, precision)
        #print("sk_precrec_score ",sk_precrec_score)
        
        
    def show_plot(self):    
        #print("Plotting ",self.cnf_matrix)
        
        cnf_matrix = self.get_conf_matrix()
        
        self.plot_dashboard(img=self.img, msk=self.msk, res=self.res, cm=cnf_matrix)
    
    def plot_dashboard(self, img, msk, res, cm, 
                              num_classes=3,
                              normalize=False, #if true all values in confusion matrix is between 0 and 1
                              title='Confusion matrix',
                              fig_width = 16.0,
                              fig_height = 12.0,
                              fig_title="Dev Dashboard",
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        fig, ax_arr = plt.subplots(2, 3, sharex=False, squeeze =False,figsize=(fig_width, fig_height), num=title)
        
        
        # msk
        ax=ax_arr[0,0]
        ax.imshow(msk)
        ax.set_title("Annotation")
        ax.axis('off')
        
        # res
        ax=ax_arr[0,1]
        ax.imshow(res)
        ax.set_title("Prediction")
        ax.axis('off')
        
        
        # image
        ax=ax_arr[1,0]
        ax.imshow(img)
        ax.set_title("Image")
        ax.axis('off')
        
        
        # conf matrix
        # -----------
        ax= ax_arr[1,1]
        ax = self.plot_conf_matrix(ax, cm)
        
        '''    
        img=ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.set_title("Confusion")
        #fig.colorbar(img, ax=ax)
           
        # set axes, labels, titles
        # ------------------------
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
        
        # prepare cm-cell-elements
        # ------------------------
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # round numbers in matrix
        cm = np.round(cm, decimals=2)
        
        # write cm-elements
        # -----------------
        thresh = cm.max() / 2. #switch black or white font
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        '''
        
        # ROC-Curve now
        # -------------
        c_roc = BM_DigitRocCurve()
        y_test = msk.ravel()
        y_pred = res.ravel()
        
        y_test = k_to_categorical(y_test, num_classes)
        y_pred = k_to_categorical(y_pred, num_classes)
        
        
        print("y_test ",y_test.shape)
        print("y_pred ",y_pred.shape)
        
        ax = ax_arr[0,2]
        ax_arr[0,2] = c_roc.calc_receiver         (y_test, y_pred=y_pred, ax=ax, caption="Pred", block=False)
        
        ax = ax_arr[1,2]
        ax_arr[1,2] = c_roc.plot_prec_recall_curve(y_test, y_pred=y_pred, ax=ax, caption="Pred", block=False)
        
        fig.tight_layout()
        fig.canvas.manager.set_window_title(fig_title)
        def onclick(event):
            print("onclick ",event)
            '''
            print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
                  ('double' if event.dblclick else 'single', event.button,
                   event.x, event.y, event.xdata, event.ydata))
            '''
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        # show plot
        # ---------
        plt.show(block=True)    
        
    def plot_conf_matrix(self, ax, conf_matrix, title="Confusion Matrix", normalize=True, decimals=3, cmap=plt.cm.Blues):
        
        ax.set_title("Confusion")
        #fig.colorbar(img, ax=ax)
           
        # set axes, labels, titles
        # ------------------------
        ax.set_title(title)
        ax.set_xlabel('Predictions \n [Non-target][Target-Organ][Tumour]' )
        ax.set_ylabel('True label  \n [Tumour][Target-Organ][Non-target]')
        ax.axis('on')
        ax.set_frame_on(True)
        
        ax.set_xticks(np.arange(conf_matrix.shape[0]+1)-.5, minor=False)
        ax.set_yticks(np.arange(conf_matrix.shape[0]+1)-.5, minor=False)
        #ax.grid(which="minor", color="b", linestyle='-', linewidth=3)
        ax.grid(which="major", color="black", linestyle='-', linewidth=1)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        # prepare cm-cell-elements
        # ------------------------
        if normalize:
            conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        # round numbers in matrix
        conf_matrix = np.round(conf_matrix, decimals=decimals)
        
        # write cm-elements
        # -----------------
        thresh = conf_matrix.max() / 2. #switch black or white font
        for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
            
            color="white" if conf_matrix[i, j] > thresh else "black"
            weight="normal"
            if i==j:
                color="red"
                weight="bold"
                
            ax.text(j, i, conf_matrix[i, j],
                     horizontalalignment="center",
                     color=color, weight=weight)
        
        img=ax.imshow(conf_matrix, interpolation='nearest', cmap=cmap)
        
        return ax 
    
    def calc_global_conf(self, data_path, num_classes=3):
        
        item_extension=['.png']
        item_sort=False
        c_files = DigitFileDir(dir_path=res_dir, item_extension=item_extension, item_sort=item_sort)
        filenames=c_files.get_file_list(item_sort=item_sort)
        for idx, filename in enumerate(filenames):
            print("filename ",idx, filename)
        
        print("files ",len(filenames))
        global_cnf= np.zeros([num_classes, num_classes])
        print("global_cnf ", global_cnf.shape)
        
        for filename in filenames:
            self.get_data(filename=filename)
            cnf_matrix = self.get_conf_matrix()
            global_cnf = np.add(cnf_matrix, global_cnf)
            print("global_cnf\n ",global_cnf)
        
        print("global_cnf\n ",global_cnf)
        print("Program stopped here")
        
        return global_cnf
        
 
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
    
    print("msk_dir ",msk_dir)
    
    
    
    item_extension=['.png']
    item_sort=False
    c_files = DigitFileDir(dir_path=res_dir, item_extension=item_extension, item_sort=item_sort)
    filenames=c_files.get_file_list()
    print("files ",len(filenames))
  
    c_scores = BM_Evaluate2D()
    c_scores.init_dirs(img_dir=img_dir,
                           msk_dir=msk_dir,
                           res_dir=res_dir)
    
    
    
    
    # load data
    _= c_scores.get_data()
    #exit()
    # calculate metrics
    #_= c_scores.calc_metrics()
    #c_scores.calc_report()
    # calculate conf
    cnf_matrix = c_scores.get_conf_matrix()
    #exit()
    #cnf_matrix = c_scores.calc_global_conf(res_dir)
    
    _ = c_scores.calc_metrics()
    #print("Program stopped here")
    #exit()
    _ = c_scores.calc_scores(cnf_matrix)
    _ = c_scores.calc_auc()
    
    c_scores.show_plot()
    
    print("Program terminated")
'''
---------------------------------------------------------------------
'''    
    
    