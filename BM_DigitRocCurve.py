'''
Created on 31 Mar 2022

@author: digit
'''
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_FILE = os.path.abspath(__file__) 

import numpy as np
import matplotlib.pyplot as plt


from BM_StartProgram import BM_StartProgram
BM_StartProgram(BASE_FILE)
'''
@Digit:
Just for sample, will be dumped
'''
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

# imports renamed for metrics
# ---------------------------
#from sklearn.metrics import roc_curve as sk_roc_curve
#from sklearn.metrics import auc as sk_auc

# used for categorising data
# --------------------------
from sklearn.preprocessing import label_binarize as sk_label_binarize

from BM_Digit_ROC import roc_curve as sk_roc_curve
from BM_Digit_ROC import auc as sk_auc
from sklearn.metrics import roc_auc_score as sk_roc_auc_score

from sklearn.metrics import f1_score as sk_f1_scrore
from sklearn.metrics import precision_recall_curve  as sk_precision_recall_curve
from sklearn.metrics import average_precision_score as sk_average_precision_score

from scipy.special import softmax

# Own imports
#from DigitProcess import set_max_priority
#set_max_priority()


class BM_DigitRocCurve(object):
    def __init__(self, num_classes=3, 
                       # categorise data
                       classes=[0, 1, 2],
                       plot_ids=[2,1,0] ):
        

        self.num_classes = num_classes
        assert num_classes == 3, "Mismatch num_classes"
        # categorise data
        self.classes=classes
        self.plot_ids = plot_ids
        
        # Parameters for figure
        # ---------------------
        self.figsize=(8,8)
        
        self.colors = ["black", "grey", "blue","lightgreen","darkgreen"]
        self.line_widths = [1,2,3,2,2]
        self.line_styles=['dotted',"solid","solid","dashed","dashed"]

        self.xlim =(0.0, 1.02)
        self.ylim =(0.0, 1.02)
        self.xlabel="False Positive Rate := 1-Sensitivity"
        self.ylabel="True Positive Rate := Specifity"
        self.legend_loc="lower center"
        
        self.class_names=["Non-target   ", 
                          "Target-organ ", 
                          "Tumour region",
                          "MICRO avg    ",
                          "MACRO avg    "]

        self.avg_line_widths = [1,2,3,2,2]
        self.avg_line_styles=['dotted','dashed',"dashed","solid","solid"]

        self.fig_title="ROC curves for classes & avg"
        
    def get_data(self):
        

        # Import some data to play with
        iris = datasets.load_iris()
        x_test = iris.data
        y_test = iris.target
        
        # Binarize the output
        print("y_test ori ",y_test.shape)
        y_test = sk_label_binarize(y_test, classes=self.classes)
        print("y_test cat ",y_test.shape)
        
        num_classes = y_test.shape[1]
        
        # Add noisy features to make the problem harder
        random_state = np.random.RandomState(0)
        n_samples, n_features = x_test.shape
        x_test = np.c_[x_test, random_state.randn(n_samples, 200 * n_features)]
        
        X=x_test
        y=y_test
        
        # shuffle and split training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
        
        # Learn to predict each class against the other
        classifier = OneVsRestClassifier(svm.SVC(kernel="linear", probability=True, random_state=random_state))
        
        
        self.num_classes = num_classes
        y_prob = classifier.fit(X_train, y_train).decision_function(X_test)
        
        # Softmax probabilities
        # ---------------------
        y_prob = softmax(y_prob, axis=1)
        
        print("y_prob ",y_prob[:10])
        #print("Program stopped here"); exit()  
        
        y_pred = np.argmax(y_prob, axis = -1)
        y_pred = sk_label_binarize(y_pred, classes=self.classes)
        
        print("y_test ", y_test.shape)
        print("y_prob ", y_prob.shape, y_prob.min(), y_prob.max())
        print("y_pred ", y_pred.shape, y_pred.min(), np.unique(y_pred), y_pred.max())
        
        # assign to class members
        # -----------------------
        self.y_test = y_test
        self.y_prob = y_prob
        self.y_pred = y_pred
        
        # Compute the average precision score
        # ...................................
        averages =(None, 'micro', 'macro')
        for average in averages:
            try: 
                avg_prec = sk_average_precision_score(y_test, y_prob, average=average)
                average = average or "classes "
                print("Prec score for ",str(average), avg_prec)
            except BaseException as e:
                print("avg_prec_score error ",str(e))    
        
        average = 'macro'
        avg_prec = sk_average_precision_score(y_test.ravel(), y_prob.ravel(), average=average)
        print("Prec score for ravel ",str(average), avg_prec)
        
        '''
        The F1 score can be interpreted as a weighted average of the precision and recall, 
        where an F1 score reaches its best value at 1 and worst score at 0. 
        
        It is the ***harmonic mean*** of precision and recall, 
        meaning the relative contribution of precision and recall to the F1 score are equal:
        '''
        averages =(None,'micro', 'macro')
        
        for average in averages:
            try:
                f1_score = sk_f1_scrore(y_test, y_pred, average=average)
                average = average or "classes "
                print("f1 score for ",str(average), f1_score)
                
                
            except BaseException as e:
                print("f1_score error ",str(e))
        average = 'macro'            
        f1_score = sk_f1_scrore(y_test.ravel(), y_pred.ravel(), average=average)
        print("f1 score for ravel ",str(average), f1_score)
            
        
        fmt='%.6lf'
        filename="y_test.txt"
        np.savetxt(filename,y_test,fmt=fmt)
        
        filename="y_pred.txt"
        np.savetxt(filename,y_pred,fmt=fmt)
        
        filename="y_prob.txt"
        np.savetxt(filename,y_prob,fmt=fmt)
        
        
        
        return(y_test, y_pred, y_prob)
    
    def set_data(self, y_test, y_pred, y_prob):
        
        self.y_test = y_test
        self.y_pred = y_pred
        self.y_prob = y_prob
        

    def calc_receiver(self, y_test, y_data, num_classes=3, ax=None, caption="", block=False, verbose=True):

        '''
        In the multiclass case, y_score corresponds to an array of  
        of probability estimates. 
        y_score has shape (n_samples, n_classes)
        The probability estimates must sum to 1 across the possible classes. 
        
        In addition, the order of the class scores must correspond to the order of labels, 
        if provided, or else to the numerical or lexicographical order of the labels in y_true. 
        
        See more information in the User guide;
        '''

        #y_test = self.y_test if y_test is None else y_test
        #y_data = self.y_prob if y_data is None else y_data
        #y_pred = self.y_pred if y_pred is None else y_pred
        num_classes = self.num_classes if num_classes is None else num_classes

        assert num_classes == 3, "Mismatch number of classes"
        
        # y_test must be categorised
        # --------------------------
        if y_test.shape[-1] != num_classes:
            '''
            @Digit: sk_label_binarize Bit too complicated to extract
            '''
            if verbose: print("binarize y_test ",y_test.shape, num_classes)
            y_test = sk_label_binarize(y_test, classes=self.classes)    
        
        assert y_test.shape[-1]==num_classes, "Mismatch category y_test"
        
        
        if verbose:
            print("Y_test ",y_test[:10])
            print("y_data ",y_data[:10])

            


        # Compute ROC curve and ROC area for each class
        dc_fpr = dict()
        dc_tpr = dict()
        dc_auc = dict()
        
        for idx in self.plot_ids:    
            # Receiver operating curve
            if verbose: print("sk_roc_curve ",idx)
            dc_fpr[idx], dc_tpr[idx], _ = sk_roc_curve(y_test[:, idx], y_data[:, idx])
            dc_auc[idx]                 = sk_auc(dc_fpr[idx], dc_tpr[idx])
            
        if verbose: print("calc micro-average ROC")
        # Compute micro-average ROC curve and ROC area
        dc_fpr["micro"], dc_tpr["micro"], _ = sk_roc_curve(y_test.ravel(), y_data.ravel())
        dc_auc["micro"]                     = sk_auc(dc_fpr["micro"], dc_tpr["micro"])
        
        if verbose: print("calc-average ROC")
        dc_fpr["macro"], dc_tpr["macro"] = self.calc_macro_avg(dc_fpr, dc_tpr)
        # calc roc_auc
        # ------------
        dc_auc["macro"] = sk_auc(dc_fpr["macro"], dc_tpr["macro"])
        
        
        ax=self.plot_receiver(dc_fpr, dc_tpr, dc_auc, ax,
                           num_classes=num_classes,
                           caption=caption)
        return ax
    
    def plot_receiver(self, dc_fpr, dc_tpr, dc_auc, ax=None,
                      num_classes=3,
                      title_pad=-28,
                      caption="", 
                      fig_title="Receiver Operat. Char & avg",
                      block=False, verbose=True):
        
        # Plot of a ROC curve for a specific class
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize, num=self.fig_title+" "+caption)
        
        
        for idx in self.plot_ids:    
        
            ax.plot(
                #data
                dc_fpr[idx],
                dc_tpr[idx],
                # line styles
                color=self.colors[idx],
                linewidth=self.line_widths[idx],
                linestyle = self.line_styles[idx],
                #label
                label="ROC %s :area = %0.2f"%(self.class_names[idx], dc_auc[idx]),)
        # Insert dummy figure to "split" legend
        # -------------------------------------    
        ax.plot([0,0],[1,1],color="white", label="-",)
        
        idx=3
        ax.plot(
            dc_fpr["micro"],
            dc_tpr["micro"],
            label="ROC %s :area = %0.2f"%(self.class_names[idx],dc_auc["micro"]),
            color=self.colors[idx],#"deeppink",
            linestyle=self.line_styles[idx],#":",
            linewidth=self.line_widths[idx]#4,
        )


        idx=4
        ax.plot(
            dc_fpr["macro"],
            dc_tpr["macro"],
            label="ROC %s :area = %0.2f"%(self.class_names[idx],dc_auc["macro"]),
            color=self.colors[idx],
            linestyle=self.line_styles[idx],
            linewidth=self.line_widths[idx])
        
        # diag line
        ax.plot([0, 1],[0, 1], color="grey", lw=1, linestyle="-.")
        
        # top line
        ax.plot([0, 1],[1, 1], color="grey", lw=1, linestyle="-.")
        
        # right line
        ax.plot([1, 1], [0, 1], color="grey", lw=1, linestyle="-.")
        
        ax.spines['top'   ].set_visible(False)
        ax.spines['right' ].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'  ].set_visible(True)
        
        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        #fig_title ="Rec. Op. Char & avg" # taken from argument
        #ax.set_title(fig_title, y=1.0, pad=title_pad)
        l2 = np.array((0.9, 0.1))
        ax.text(*l2, fig_title, fontsize=16,
              rotation=90, rotation_mode='anchor',
              transform_rotates_text=True)
        #ax.set_title('Manual y', y=1.0, pad=-14)
        ax.legend(prop={'family': 'DejaVu Sans Mono'}, loc=self.legend_loc)
        
        if verbose:
            for idx in range(num_classes):
        
                print("dc_fpr ",idx, dc_fpr[idx].shape, dc_fpr[idx].dtype, dc_fpr[idx])
                print("dc_tpr ",idx, dc_tpr[idx].shape, dc_tpr[idx].dtype, dc_tpr[idx])
            key='micro'    
            print("dc_fpr ",key, dc_fpr[key].shape, dc_fpr[key].dtype)
            print("dc_tpr ",key, dc_tpr[key].shape, dc_tpr[key].dtype)
        
            key='macro'    
            print("dc_fpr ",key, dc_fpr[key].shape, dc_fpr[key].dtype)
            print("dc_tpr ",key, dc_tpr[key].shape, dc_tpr[key].dtype)
        
        plt.show(block=block)
        return ax
    
    def plot_prec_recall_curve(self, y_test, y_data, 
                               ax=None, 
                               title_pad=-28,
                               num_classes=3, caption="", 
                               fig_title = "Precicion Recall Curve",
                               block=False, verbose=True):
        
        num_classes = self.num_classes if num_classes is None else num_classes

        dc_prec = dict()
        dc_rcall = dict()
        dc_auc = dict()

        for idx in self.plot_ids:    
           
            # Prec recall curve (is just an offspin)
            if verbose: print("calc PrecRec ",idx)
            dc_prec[idx], dc_rcall[idx], _ = sk_precision_recall_curve( y_test[:, idx], y_data[:, idx])
            dc_auc[idx]                    = sk_average_precision_score(y_test[:, idx], y_data[:, idx])
        
        
        # A "micro-average": quantifying score on all classes jointly
        if verbose: print("calc PrecRec micro ",idx)
        
        dc_prec["micro"], dc_prec["micro"], _ = sk_precision_recall_curve( y_test.ravel(), y_data.ravel())
        dc_auc["micro"]                     = sk_average_precision_score(y_test, y_data, average="micro")
        print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(dc_auc["micro"]))
        
        #dc_prc["macro"], dc_rec["macro"], _ = sk_precision_recall_curve( y_test, y_data)
        #dc_prc_auc["macro"]                 = sk_average_precision_score(y_test, y_data, average="macro")
        #print('Average precision score, Macro-averaged over all classes: {0:0.2f}'.format(dc_prc_auc["macro"]))
        
        if verbose: print("calc PrecRec macro ",idx)
        # A "micro-average": quantifying score on all classes jointly
        dc_prec["macro"], dc_rcall["macro"] = self.calc_macro_avg(dc_prec, dc_rcall)
        try:
            dc_auc["macro"]              = sk_average_precision_score(y_test, y_data, average="macro")
        except BaseException as e:
            print("sk_average_precision_score Error ",str(e))
            dc_auc["macro"] = 0         
        
        # Plot of a ROC curve for a specific class
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize, num="Precision recall curve"+" "+caption)

        for idx in self.plot_ids:    
        
            ax.plot(
                #data
                dc_rcall[idx],
                dc_prec[idx],
                # line styles
                color=self.colors[idx],
                linewidth=self.line_widths[idx],
                linestyle = self.line_styles[idx],
                #label
                label="Prec Recall %s :area = %0.2f"%(self.class_names[idx], dc_auc[idx]),)
        # Plot dummy for empty line in label
        ax.plot([0,0],[1,1],color="white", label="-",)    
        
        idx=3
        '''
        plt.step(dc_rcall['micro'], 
                 dc_prec['micro'], 
                 where='post', 
                 color=self.colors[idx],
                 linestyle=self.line_styles[idx],
                 linewidth=self.line_widths[idx],
                 label="Prec Recall %s :area = %0.2f"%(self.class_names[idx], dc_auc['micro']),)
        
        '''
        idx=4
        ax.step(dc_rcall['macro'], 
                 dc_prec['macro'], 
                 where='post', 
                 color=self.colors[idx],
                 linestyle=self.line_styles[idx],
                 linewidth=self.line_widths[idx],
                 label="Prec Recall %s :area = %0.2f"%(self.class_names[idx], dc_auc['macro']),)
        
        # top line
        ax.plot([0, 1],[1, 1], color="grey", lw=1, linestyle="-.")
        # right line
        ax.plot([1, 1], [0, 1], color="grey", lw=1, linestyle="-.")
        
        f_scores = (0.33,0.5,0.66)
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            line, = plt.plot(x[y >= 0], y[y >= 0], color='gray', linestyle='dashed', alpha=0.2)
            ax.annotate('f1={0:0.2f}'.format(f_score), xy=(0.9, y[45] + 0.02))#Digit: new
        
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        
        ax.set_xlim(self.xlim) 
        ax.set_ylim(self.ylim)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.set_title(fig_title, y=1.0, pad=title_pad)
        
        l2 = np.array((0.1, 0.1))
        ax.text(*l2, fig_title, fontsize=16,
              rotation=90, rotation_mode='anchor',
              transform_rotates_text=True)
        
        ax.legend(prop={'family': 'DejaVu Sans Mono'}, loc=self.legend_loc)
        
        plt.show(block=block)
        
        return ax
    
    
    def calc_macro_avg(self, x_data, y_data, num_classes=3, caption=""):
        
        num_classes = self.num_classes if num_classes is None else num_classes
        
        # ..........................................
        # Compute macro-average ROC curve and ROC area
        
        # First aggregate all false positive rates
        # Join a sequence of arrays along an existing axis.
        all_fpr = np.unique(np.concatenate([x_data[i] for i in range(num_classes)]))
        
        # Then interpolate all ROC curves at those points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(num_classes):
            
            # One-dimensional linear interpolation for monotonically increasing sample points.
    
            # Returns the one-dimensional piecewise linear interpolant to a function
            # with given discrete data points (`xp`, `fp`), evaluated at `x`.
            mean_tpr += np.interp(all_fpr, x_data[i], y_data[i])
        
        # Finally average it and compute AUC
        mean_tpr /= num_classes
        
        return all_fpr, mean_tpr


'''
-----------------------------------------------------
'''    
if __name__ == '__main__': 
    c_roc = BM_DigitRocCurve()
    y_test, y_pred, y_prob=c_roc.get_data()
    #c_roc.roc_single(y_test, y_data=y_pred, caption="Pred")
    #c_roc.roc_single    (y_test, y_data=y_prob, caption="Prob")
    
    c_roc.calc_receiver(y_test, y_data=y_pred, caption="Pred", block=False)
    c_roc.calc_receiver(y_test, y_data=y_prob, caption="Prob", block=False)
    #plt.show(block=True)
    
    c_roc.plot_prec_recall_curve(y_test, y_data=y_pred, caption="Pred", block=False)
    c_roc.plot_prec_recall_curve(y_test, y_data=y_prob, caption="Prob", block=False)
    plt.show(block=True)
    
    print("Program terminated")
