'''
Created on 14 Jun 2022

@author: digit
'''
import os
import numpy as np
import time
#from numba import cuda

import tensorflow as tf
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

# Own imports
# -----------
# from DigitNvidia import set_gpu#CPU/GPU
# set_gpu(True)
# from DigitProcess import set_max_priority
# set_max_priority()

# Own imports
# -----------
import DigitConfig

''' Config Json File '''
conf = DigitConfig.Config('_BostonMedical')

from BM_ImagePil_dkube import load_img, load_mask

def get_data( img_dir, msk_dir, res_dir, filename = None):
    filename= 'Abyssinian_17' if filename is None else filename
    
    filename = os.path.basename(filename)
    filename = os.path.splitext(filename)[0]
    
    img_file = os.path.join(img_dir, filename+'.jpg')
    msk_file = os.path.join(msk_dir, filename+'.png')
    res_file = os.path.join(res_dir, filename+'.png')
    
    
    print("Load data ", filename)
    img = load_img(img_file)
    msk = load_mask(msk_file)
    res = load_mask(res_file)
    print("Data loaded ",filename)
    return img, msk, res

# DIGIT: base code to be transerred to CUDA
# -----------------------------------------
def cpu_conf_matrix_from_image(m_test, m_pred, num_classes=3):
    cnf = np.zeros((num_classes, num_classes),dtype=np.uint32)
    # m_test, m_pred are images and matrices
    # --------------------------------------
    for icol in range(m_test.shape[0]):
        for irow in range(m_test.shape[1]):
            idx_col = m_test[icol][irow]
            idx_row = m_pred[icol][irow]
            cnf[idx_col][idx_row]+=1
                
    return cnf

# DIGIT: base code to be transerred to CUDA
# ------------------------------------------
def cpu_conf_matrix(y_test, y_pred, num_classes=3):
    # DIGIT: base code to be transerred into CUDA
    # -------------------------------------------
    cnf = np.zeros((num_classes, num_classes),dtype=np.uint32)
    
    for idx in range(y_test.shape[0]):
        idx_col = y_test[idx]
        idx_row = y_pred[idx]
        cnf[idx_col][idx_row]+=1 #--> atomize this for CUDA
                
    return cnf

def cuda_conf_matrix(y_test, y_pred, num_classes=3):
    
    # ravel data to 1dim arrays
    if len(y_test.shape)>1:
        y_test = y_test.ravel()
    if len(y_pred.shape)>1:
        y_pred = y_pred.ravel()
    
    d_y_test = cuda.to_device(y_test) # Copy of x on the device
    d_y_pred = cuda.to_device(y_pred) # Copy of y on the device
    d_conf   = cuda.device_array(shape=(num_classes, num_classes), dtype=np.uint32) # Take care of shape=(n,)    
        
    threads_per_block = 128
    blocks_per_grid = 32
    
    cuda_core_conf_matrix[blocks_per_grid, threads_per_block](d_y_test, d_y_pred, d_conf)
    cuda.synchronize
    
    gpu_cnf_matrix = d_conf.copy_to_host()
    return gpu_cnf_matrix
'''
#Process time conf_gpu 0.368465667 [sec]
@cuda.jit
def cuda_core_conf_matrix(y_test, y_pred, cnf):
    # DIGIT: base code to be transerred into CUDA
    # -------------------------------------------
    idx = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(idx, y_test.shape[0], stride):
        idx_col = y_test[i]
        idx_row = y_pred[i]
      
        #cnf[idx_col][idx_row] +=1 #--> atomize this for CUDA
        #cuda.atomic.add(cnf, (y_test[i], y_pred[i]), 1) #Process time conf_gpu 0.380662373 [sec]
        cuda.atomic.add(cnf, (idx_col, idx_row), 1)      #Process time conf_gpu 0.387899054 [sec]
'''
#0.381820359
#0.384471093
    
'''
--------------------
'''        
if __name__ == '__main__': 
    num_classes=3
    verbose=True
    loops = 100#0
    # reset gpu memory
    # ----------------
    device = cuda.get_current_device()
    device.reset()
    os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
    #nvidia-smi -L

    gpus = cuda.list_devices()
    for idx, gpu in enumerate(gpus):
        print("gpu ",idx, gpu)
    print("cuda.detect         ",cuda.detect())
    print("cuda.is_available() ",cuda.is_available())
    y_test = np.asarray([0,1,2,0,1,2, 0, 1, 2, 1 ])
    y_pred = np.asarray([0,1,2,0,1,2, 1, 2, 1, 0 ])


    img_dir = conf["IMG_DIR"] 
    msk_dir = conf["MASK_DIR"]
    res_dir = conf["RESULT_DIR"]

    img, msk,res = get_data(img_dir, msk_dir, res_dir )
    y_test = msk.ravel()
    y_pred = res.ravel()
    if verbose:
        print("y_test ", y_test.shape, y_test.min(), np.unique(y_test), y_test.max())
        print("y_pred ", y_pred.shape, y_pred.min(), np.unique(y_pred), y_pred.max())
    
    assert y_test.shape[:-1] == y_pred.shape[:-1], "Mismatch shape y_test vs y_pred"


    # Digit conf on cpu
    # -----------------
    time_start = time.perf_counter()
    for _ in range(loops):
        cpu_cnf_matrix = cpu_conf_matrix(y_test, y_pred)
    time_end = time.perf_counter()
    if verbose: print("Process time conf_cpu %2.9lf [sec]"%float(time_end-time_start))
    print("cpu_cnf_matrix \n ", cpu_cnf_matrix)
    
    strides=16
    coalesced =True #Process time conf_gpu 0.386127246 [sec]
    coalesced =False #DIGIT: Does not work
    
      
        
    threads_per_block = 128
    blocks_per_grid = 32
    time_start = time.perf_counter()
    
    
    for _ in range(1):
        d_y_test = cuda.to_device(y_test) # Copy of x on the device
        d_y_pred = cuda.to_device(y_pred) # Copy of y on the device

        d_conf   = cuda.device_array(shape=(num_classes, num_classes), dtype=np.float32) # Take care of shape=(n,)  
        cuda_core_conf_matrix[blocks_per_grid, threads_per_block](d_y_test, d_y_pred, d_conf)
        cuda.synchronize
    
    time_end = time.perf_counter()
    gpu_cnf_matrix = d_conf.copy_to_host()
    gpu_cnf_matrix = np.asarray(gpu_cnf_matrix, dtype=np.int32)
    

    if verbose: print("Process time conf_gpu %2.9lf [sec]"%float(time_end-time_start))
    
    print("gpu_cnf_matrix \n ", gpu_cnf_matrix)
    
    
    # conf from sklearn
    # -----------------
    time_start = time.perf_counter()
    for _ in range(loops):
        sk_cnf_matrix = (sk_confusion_matrix(y_test, y_pred))
    time_end = time.perf_counter()
    if verbose: print("Process time sklean   %2.9lf [sec]"%float(time_end-time_start))
    
    # conf from tensorflow
    # --------------------
    time_start = time.perf_counter()
    for _ in range(loops):
        tf_cnf_matrix = tf.math.confusion_matrix(y_test, y_pred, dtype=tf.dtypes.int32, num_classes=num_classes)
        tf_cnf_matrix = tf_cnf_matrix.numpy()
    time_end = time.perf_counter()
    if verbose: print("Process time tensor   %2.9lf [sec]"%float(time_end-time_start))
    
    
    gpu_cnf_matrix = cuda_conf_matrix(y_test, y_pred)
    
    
    
    # check result
    # ------------
    assert np.allclose(sk_cnf_matrix, tf_cnf_matrix), "Mismatch sk vs. tf"
    assert np.allclose(sk_cnf_matrix, cpu_cnf_matrix), "Mismatch sk vs cuda_core"
    assert np.allclose(sk_cnf_matrix, gpu_cnf_matrix), "Mismatch sk vs cuda_call"

    
    print("Program terminated")
    
    '''
    loops=1000
    
    
    Process time conf_cpu 58.495308644 [sec]

    Process time conf_gpu  0.430640554 [sec]
    Process time sklean    2.033303041 [sec]
    Process time tensor    3.934432860 [sec]
    
    '''
    
    
'''
--------------------------------------------------------
'''    
    
                 