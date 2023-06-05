'''
Created on Feb 23, 2023

@author: digit

https://neptune.ai/blog/tensorboard-tutorial

pip install -U tensorboard-plugin-profile

tensorboard dev upload --logdir {logdir}
'''
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_FILE = os.path.abspath(__file__)

import numpy as np
from time import gmtime, strftime
import logging
import io
import matplotlib.pyplot as plt

# Init logging
format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=print, datefmt="%H:%M:%S")

# import tf and Keras
# -------------------
import tensorflow as tf
from tensorflow.keras.utils import to_categorical as k_to_categorical

# Used for Tensorboard pcl
import open3d as o3d
from open3d.visualization.tensorboard_plugin import summary as o3d_summary
from open3d.visualization.tensorboard_plugin.util import to_dict_batch as o3d_to_dict_batch


# Own imports
# -----------
from DigitColours import colours
from BM_Evaluate2D import BM_Evaluate2D
from BM_DigitRocCurve import BM_DigitRocCurve

class DigitKerasTensorboard(object):
    def __init__(self):
        pass
 
    def eval_summary_init_writers(self, log_dir):                
        log_dir = os.path.join(self.log_dir, "_tf_log")
        log_dir = os.path.join("/media/BostonNFSAdele/dkube/dkube/users/sahilwadhwa19/model", "_tf_log")
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)
        if os.path.isdir(log_dir):    
            print("Init model: writers_log_dir confirmed %s "%log_dir) 
        else:
            logging.error("Init model: writers_log:dir cannot be created %s "%log_dir) 
            return []
        
        writers =[]
        try:
            self.file_writer_cm    = tf.summary.create_file_writer(log_dir + '/cm',   name="BM_conf_matrix"); writers.append(self.file_writer_cm )
            print(self.file_writer_cm)
            
            self.file_writer_cm_n  = tf.summary.create_file_writer(log_dir + '/cm_n', name="BM_conf_matrix_norm"); writers.append(self.file_writer_cm_n)

            self.file_writer_img =  tf.summary.create_file_writer(log_dir + '/img', name ="BM_images" ); writers.append(self.file_writer_img)
            self.file_writer_seg =  tf.summary.create_file_writer(log_dir + '/seg', name ="BM_segment"); writers.append(self.file_writer_seg)
            self.file_writer_pred = tf.summary.create_file_writer(log_dir + '/pred',name ="BM_preds");   writers.append(self.file_writer_pred)
            self.file_writer_prob = tf.summary.create_file_writer(log_dir + '/prob',name ="BM_probs");   writers.append(self.file_writer_prob)
     
            self.file_writer_roc = tf.summary.create_file_writer(log_dir + '/roc',  name="BM_ROC_curves"); writers.append(self.file_writer_roc)
            self.file_writer_prc = tf.summary.create_file_writer(log_dir + '/prc',  name="BM_PRC_curves"); writers.append(self.file_writer_prc)

            # Metrics 
            self.file_writer_met = tf.summary.create_file_writer(log_dir + '/met',name="BM_metrics");    writers.append(self.file_writer_met)
            
            # Tensorboard pointcloud
            self.file_writer_pcl = tf.summary.create_file_writer(log_dir + '/pcl',name="BM_Pointcloud"); writers.append(self.file_writer_pcl)

            
            # Profiling - not in list            
            self.file_writer_tf_graph = tf.summary.create_file_writer(log_dir + '/tf_graph',name="BM_Profiling"); #writers.append(self.file_writer_tf_graph)

            # Tensorboard profiler
            # --------------------
            # tf.summary.trace_on(graph=True, profiler=True)

        except BaseException as e:
            logging.error("Error creating filewriters %s "%str(e))

        #self.writers = writers
        return writers    
 
    def eval_summary_flush_writers(self)->bool:
        try:
            for writer in self.writers:
                writer.flush()
        except BaseException as e:
            logging.error("Writers flush error %s "%str(e))
            return False
    
    def eval_summary_close_writers(self, writers=None)->bool:
        
        writers = writers or self.writers
        try:
            for writer in writers:
                writer.flush()
                writer.close()
        except BaseException as e:
            logging.error("Train end close writer error %s "%e)
            return False
        print("Writers flushed and closed %d"%len(writers))
        return True
    
    
    def eval_plot_init_figure(self, rows=1, cols=1, fig_width=8, fig_height=8, title="BM-chart"):    
        fig, ax_arr = plt.subplots(rows, cols, sharex=False, squeeze=False, 
                                   figsize=(fig_width, fig_height), num=title)
        return fig, ax_arr

    def eval_plot_to_image(self, figure=None, use_colour=False):
        
        if figure is None:
            figure,_=self.eval_plot_init_figure(title="BM-Empty")
        """Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call."""
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        
        # DIGIT Memo: image already expanded by batchsize
        # ----------------------------------------------- 
        image = tf.expand_dims(image, 0)
        return image
    
    def eval_summary_conf(self, conf_matrix, epoch, normalize=False, title="Confusion Matrix discrete")->bool: 
        if normalize:
            title="Confusion Matrix normalize"
                
        fig, ax_arr = self.eval_plot_init_figure(title=title)
        ax = ax_arr[0,0]
        # c_conf = BM_Evaluate2D(self.img_dir, self.mask_dir, self.res_dir)
        c_conf = BM_Evaluate2D()
        ax = c_conf.plot_conf_matrix(ax, conf_matrix, normalize=normalize)
        fig.tight_layout()
        #fig.canvas.manager.set_window_title(fig_title)
        
        cm_image = self.eval_plot_to_image(fig)
        # Log the confusion matrix as an image summary.
        
        if normalize: writer = self.file_writer_cm_n 
        else: writer= self.file_writer_cm
        
        try:
            with writer.as_default():
                tf.summary.image("BM_Epoch_Confusion_Matrix", cm_image, step=epoch)
                # log flattened conf_matrix to histogram
                tf.summary.histogram(name=title, data=np.diag(conf_matrix), step=epoch, buckets=None, description=title)

        except BaseException as e:
            logging.error("eval_summary_conf error %s "%str(e))
            return False
        return True        
    
    def eval_summary_metrics(self, y_test, y_pred, epoch, num_classes=3, verbose=True)->bool:
        # calculate ROC
        # -------------
        c_roc = BM_DigitRocCurve()
        c_roc.c
        
            
    def eval_summary_curves(self, y_test, y_pred, y_prob, epoch, num_classes=3, verbose=True)->bool:
        
        if not hasattr(self, "file_writer_roc"):
            return False
        
        if not hasattr(self, "file_writer_prc"):
            return False
        
        # Prepare data
        # ------------
        if verbose:
            print("Summary curves: y_test ",y_test.shape, y_test.dtype)
            print("Summary curves: y_pred ",y_pred.shape, y_pred.dtype)
        assert y_test.shape == y_pred.shape, "Mismatch y_test.shape == y_pred.shape"
        
        # ravel data to 1dim arrays
        # -------------------------
        y_test = y_test.ravel()
        y_pred = y_pred.ravel()
        y_prob = y_prob.ravel()
        
        # Categorize input data using Keras
        # ---------------------------------
        y_test = k_to_categorical(y_test, num_classes)
        y_pred = k_to_categorical(y_pred, num_classes)
        y_prob = k_to_categorical(y_prob, num_classes)
        
        # calculate ROC
        # -------------
        c_roc = BM_DigitRocCurve()
        
        fig, ax_arr = self.eval_plot_init_figure(title="ROC Curve Pred")
        ax = ax_arr[0,0]
        ax = c_roc.calc_receiver(y_test, y_data=y_pred, ax=ax, caption="Pred", block=False)
        fig.tight_layout()
        # convert to image
        # ----------------
        roc_image_pred = self.eval_plot_to_image(fig)

        '''
        fig, ax_arr = self.eval_plot_init_figure(title="ROC Curve Prob")
        ax = ax_arr[0,0]
        ax = c_roc.calc_receiver(y_test, y_pred=y_prob, ax=ax, caption="Prob", block=False)
        fig.tight_layout()
        # convert to image
        # ----------------
        roc_image_prob = self.eval_plot_to_image(fig, use_colour=True)
        '''
        # save images
        # -----------
        try:
            with self.file_writer_roc.as_default():
                tf.summary.image("BM_ROC_Curve_Pred", roc_image_pred, step=epoch, description="Receiver Operating Curve for predictions")
                #tf.summary.image("BM_ROC_Curve_Prob", roc_image_prob, step=epoch, description="Receiver Operating Curve for probabilites")
        except BaseException as e:
            logging.error("eval_summary_ROC error %s "%str(e)) 
            return False       
    
        fig, ax_arr = self.eval_plot_init_figure(title="PRC Curve Pred")
        ax = ax_arr[0,0]
        ax = c_roc.plot_prec_recall_curve(y_test, y_data=y_pred, ax=ax, caption="Pred", block=False)
        fig.tight_layout()
        # convert to image
        # ----------------
        prc_image_pred = self.eval_plot_to_image(fig)

        '''
        fig, ax_arr = self.eval_plot_init_figure(title="PRC Curve Prob")
        ax = ax_arr[0,0]
        ax = c_roc.plot_prec_recall_curve(y_test, y_pred=y_prob, ax=ax, caption="Prob", block=False)
        fig.tight_layout()
        '''
        # convert to image
        # ----------------
        #prc_image_prob = self.eval_plot_to_image(fig)
        
        # save images
        # -----------
        try:
            with self.file_writer_prc.as_default():
                tf.summary.image("BM_Prec_Recall_Curve_Pred", prc_image_pred, step=epoch, description="Precicion Recall Curve for predictions")
                #tf.summary.image("BM_Prec_Recall_Curve_Prob", prc_image_prob, step=epoch, description="Precicion Recall Curve for probabilities")
        except BaseException as e:
            logging.error("eval_summary_ROC error %s "%str(e))
            return False
        return True        
    
    def eval_concat_list(self, items, axis=0, verbose=True):
        # Concatenate items on first axis
        # -------------------------------
        try:
            for idx, item in enumerate(items):
                if idx==0:
                    conc = item
                else:
                    conc = np.concatenate((item, conc), axis=axis) 
            if verbose:
                print("eval_concat_list ", conc.shape, len(conc.shape), conc.min(), conc.max(), conc.dtype)
            return conc
        except BaseException as e:
            print("idx  ",idx)
            print("item ",item.shape)
            print("conc ",conc.shape)
            logging.error("eval_concat_list error %s ",e)
            return []
    
    def eval_summary_img(self, items, names, epoch, title="BM_Images", description=None, max_outputs=32)->bool: 
        #instead of iterating, concatenate 4 images such that tn_logits 
        # and t_label will have shape of [4, h, w, 1]
        
        #https://stackoverflow.com/questions/51163871/tensorboard-image-summaries
        
        #im_summary = tf.Summary.Image(encoded_image_string=im_bytes)
        #im_summary_value = [tf.Summary.Value(tag=self.confusion_matrix_tensor_name, 
        #                                     image=im_summary)]
        
        # Concatenate results
        # -------------------
        conc = self.eval_concat_list(items)
        
        # write results
        # -------------
        try:
            with self.file_writer_img.as_default():
                tf.summary.image(name=title, data=conc, step=epoch, max_outputs=max_outputs)
        except BaseException as e:
            logging.error("eval_summary_img Error %s "%str(e))
            return False
        return True        

    def eval_summary_seg(self, items, names, epoch, title="BM_Segmentations", description=None, max_outputs=32)->bool: 
        # Concatenate results
        # -------------------
        conc = self.eval_concat_list(items)
        print("eval_summary_prob ", conc.shape)
        
        
        # write results
        # -------------
        try:
            with self.file_writer_seg.as_default():
                tf.summary.image(name=title, data=conc, step=0, max_outputs=max_outputs)#static
        except BaseException as e:
            logging.error("eval_summary_seg Error %s "%str(e))
            return False
        return True
        
    def eval_summary_prob(self, items, names, epoch, title="BM_Probabilities", description=None, max_outputs=32)->bool: 
        
        # Concatenate results
        # -------------------
        conc = self.eval_concat_list(items)
        print("eval_summary_prob ", conc.shape)
        
        # write results
        # -------------
        try:
            with self.file_writer_prob.as_default():
                tf.summary.image(name=title, data=conc, step=epoch, max_outputs=max_outputs)
        except BaseException as e:
            logging.error("eval_summary_prob Error %s "%str(e))
            return False
        return True
        
    def eval_summary_pred(self, items, names, epoch, title="BM_Predictions", description=None, max_outputs=32)->bool: 
        
        # Concatenate results
        # -------------------
        conc = self.eval_concat_list(items)
        # write results
        # -------------
        try:
            with self.file_writer_pred.as_default():
                tf.summary.image(name=title, data=conc, step=epoch, max_outputs=max_outputs)
        except BaseException as e:
            logging.error("eval_summary_pred Error %s "%str(e))
            return False
        return True

    def eval_summary_pcl(self, epoch):
        
        try:
            cube = o3d.geometry.TriangleMesh.create_box(1, 2, 4)
            cube.compute_vertex_normals()
            colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
            
            logdir = self.file_writer_pcl.get_logdir()
            with self.file_writer_pcl.as_default():
                cube.paint_uniform_color(colors[epoch])
                o3d_summary.add_3d('cube',
                               o3d_to_dict_batch([cube]),
                               step=epoch,
                               logdir=logdir)

        except BaseException as e:
            logging.error("eval_summary_pcl Error %s "%str(e))            


'''
--------------------------------------------
'''        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    