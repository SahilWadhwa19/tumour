'''
Created on 31.01.2021

@author: DIGIT
'''
import os
import numpy as np

import tensorflow
import tensorflow.keras

from BM_ImagePil_dkube import load_img 
from BM_ImagePil_dkube import load_mask
from BM_ImagePil_dkube import save_mask
from BM_CheckPaths_dkube import check_paths_synch


class BM_DataGen(tensorflow.keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, 
                 # input paths have to be set
                 # --------------------------
                 img_paths, 
                 mask_paths,
                 # train data
                 batch_size=1, 
                 #input data
                 img_size=(512, 512), 
                 num_channels=1,
                 num_slices=0, #DIGIT not used in orginal DataGen
                 mask_type = np.int16, #DigitDebug
                 name="DataGen",        
                 # Data augmentation
                 data_augment=False):  
        
        # training 
        self.batch_size = batch_size
        # input data
        self.img_size = img_size 
        self.num_channels = num_channels
        self.num_slices = num_slices
        
        self.img_shape = self.img_size + (num_channels,)
        self.name = name
        self.data_augment = data_augment
        self.mask_type = mask_type
        
        if self.num_channels==1:
            self.color_mode = 'grayscale'
            print("Load images in grayscale for img_size ",self.img_size)
        else:
            self.color_mode = 'rgb'
            print("Load images in rgb for img_size ",self.img_size)
        
        # Input paths have to be given
        # ----------------------------
        self.img_paths  = img_paths
        self.mask_paths = mask_paths

        print("UN_DataGen loaded %d img_paths  "% len(self.img_paths))
        print("UN_DataGen loaded %d mask_paths "% len(self.mask_paths))
        if len(self.img_paths) != len(self.mask_paths):
            raise ValueError("UN_DataGen Error number of images and masks do not match")
        
        assert check_paths_synch(self.img_paths, self.mask_paths)
        if data_augment:
            self.c_transform_affine = BM_TransformAffine()
            self.c_transform_nonaffine = BM_TransformNonAffine()

    
    # Object must have a len
    # ----------------------    
    def __len__(self):
        if len(self.img_paths) != len(self.mask_paths):
            raise ValueError("UN_DataGen Error number of images and masks do not match")
        ret = len(self.mask_paths) // self.batch_size
        assert ret > 0
        return ret 
    

    # Load images
    # -----------
    def x(self, load_all=False, maxitems= np.iinfo(np.uint32).max):
        
        # maxitems used here
        num_items = min(len(self)*self.batch_size, maxitems)
        print("Load %d images for %d batches"%(num_items, len(self)))
        
        x = np.zeros((num_items,) + self.img_size + (self.num_channels,), dtype="float32")*999
        
        for idx in range(num_items):
            
            # load image for x
            # ----------------
            img = load_img(self.img_paths[idx], color_mode=self.color_mode, target_size=self.img_size)
            img = np.asarray(img)
 
            """ added """
            if self.num_channels==1:
                img = np.expand_dims(img, 2)
            
            assert img.shape == self.img_shape
            
            #print("img.shape num_channels ",img.shape, self.num_channels)
            #assert (np.asarray(img)).shape[2]==self.num_channels
            x[idx] = img
        
        # return an array [num_items, size, size, num_items, num_channels]
        # ----------------------------------------------------------------
        assert x.min()>=0
        assert x.max()<=255
        return x
    
    # get filenames of images
    # -----------------------
    def x_files(self, load_all=False, maxitems= np.iinfo(np.uint32).max):
    
        # maxitems used here
        num_items = min(len(self)*self.batch_size, maxitems) 
        print("Load %d x_files images for %d batches"%(num_items, len(self)))
    
        files=[]
        for idx in range(num_items):
            files.append(os.path.basename(self.img_paths[idx]))
            
        # return a list
        # -------------
        return files    
    
    def get_x(self, idx):
        # Returns an image for and index and also returs filename
        # -------------------------------------------------------
        
        if idx >= len(self.img_paths):
            return None, None
        
        try:
            # load image for idx
            # ------------------
            filename = os.path.basename(self.img_paths[idx])
            img = load_img(self.img_paths[idx], color_mode=self.color_mode, target_size=self.img_size)
            """ added """
            if self.num_channels==1:
                img = np.expand_dims(img, 2)
            
            assert img.shape == self.img_shape
            assert img.min()>=0 
            assert img.max()< 256
            
            return img, filename
        except BaseException as e:
            print("get_x error ",str(e))
            return None, None    
    
    def get_y(self, idx):
        # Returns a mask for and index and also returs filename
        # -----------------------------------------------------
        if idx >= len(self.mask_paths):
            return None, None
        
        try:
            # Load mask for y
            # ---------------
            filename = os.path.basename(self.mask_paths[idx])
            
            mask = load_mask(self.mask_paths[idx], dtype=self.mask_type, target_size=self.img_size)
            assert mask.shape == self.img_size
            assert mask.min()>=0 
            assert mask.max()<=2
            
            return mask, filename
        except BaseException as e:
            print("get_y error ",str(e))
            return None, None    
    
    
    # load masks
    # ----------
    def y(self, load_all=False, maxitems= np.iinfo(np.uint32).max, verbose=False):
        
        # maxitems used here
        num_items = min(len(self)*self.batch_size, maxitems)
        # load data for batches
        # ---------------------
        print("Load %d y_masks for %d batches"%(num_items, len(self)))
        
        y = np.ones((num_items,) + self.img_size, dtype=self.mask_type)*9
        try:
            for idx in range(num_items):
                
                # Load mask for y
                # ---------------
                mask = load_mask(self.mask_paths[idx], dtype=self.mask_type, target_size=self.img_size, verbose=verbose)
                
                
                # Expand mask dimension
                # This converts shape (size,size) to shape (size, size,1)
                #if self.num_channels==1:
                #if True:
                    #mask = np.expand_dims(mask, 2)
                
                # Check shape and values
                # ----------------------
                if mask.shape[:2] != self.img_shape[:2]:
                    print( "mask.shape ",mask.shape)
                    print( "img_shape  ", self.img_shape)
                    raise ValueError("DataGen get_y assert shape ",mask.shape, self.img_shape)

                assert mask.min()>=0 and mask.max()<=2
                
                # Expand the mask here
                # --------------------
                # mask = np.expand_dims(mask, 2)
                y[idx] = mask
    
                # return an array [num_items, size, size]
                # ---------------------------------------
            unique, count = np.unique(y,return_counts=True)
            print("DataGen get_y ",y.shape, y.min(), unique, count, y.max())
            assert y.min()>=0 
            assert y.max()<=2
        
        except BaseException as e:
            raise ValueError("DataGen get_y error ",str(e))    
        
        #print("DataGen get_y ",y.shape, y.min(), unique, count, y.max())
 
        # return an array [num_items, size, size]
        # ---------------------------------------
        return y
    
    # get filenames of masks
    # ----------------------
    def y_files(self, load_all=False, maxitems= np.iinfo(np.uint32).max):
        # maxitems used here
        num_items = min(len(self)*self.batch_size, maxitems)
        print("Load %d y_files for %d batches"%(num_items, len(self)))    
        files=[]
        for idx in range(num_items):
            files.append(os.path.basename(self.mask_paths[idx]))
        # return a list
        # -------------
        return files    
    
    # The magic method __getitem__ is basically used for accessing list items, 
    # dictionary entries, array elements etc. It is very useful for a quick lookup of instance attributes. 
    # ------------
    def __getitem__(self, idx):
        """Returns tuple (image, mask) correspond to batch #idx."""
        
        # index for corresponding slices
        # ------------------------------
        i = idx * self.batch_size
        verbose = False
        assert idx >= 0 and self.batch_size > 0 
        if verbose: print(" %s get_item batch = %d of %d for batch %d"%(self.name, idx, len(self), self.batch_size))
        
        
        assert len(self.img_paths )== len(self.mask_paths)
            
        batch_img_paths  = self.img_paths [i : i + self.batch_size]
        batch_mask_paths = self.mask_paths[i : i + self.batch_size]
        
        assert check_paths_synch(batch_img_paths, batch_mask_paths)
        
        
        
        
        # x: images -->dtype="float32" 
        # ----------------------------
        x = np.ones((self.batch_size,) + self.img_size + (self.num_channels,), dtype="float32")*9999
        
        # y: masks (one channel only)
        # ---------------------------
        y = np.ones((self.batch_size,) + self.img_size + (1,), dtype=self.mask_type)*9


        # Planned for 3d:
        # x =np.ones((self.batch_size,) + (self.num_slices,)+ [...]
        # y =np.ones((self.batch_size,) + (self.num_slices,)+ [...]
        # ---------------------------------------------------------
        
        # Augmentation: init lists
        # ------------------------
        if self.data_augment:
            img_list=[] 
            msk_list=[]
           
        for idx in range(self.batch_size):
            img_path  = batch_img_paths [idx]
            mask_path = batch_mask_paths[idx]
            
            assert check_paths_synch([img_path], [mask_path])
            
            #print("DataGen Load ",os.path.basename(img_path), os.path.basename(mask_path))
            
            # debug
            #filename = os.path.basename(path)
            #print("getitem load %d %s"%(idx, filename))
            
            # load image for x with colormode defined by num_channels
            # -------------------------------------------------------
            img = load_img(img_path, color_mode=self.color_mode, target_size=self.img_size)
            
            # img as array
            img = np.asarray(img)
            # print("img.shape num_channels ",idx, self.color_mode, img.shape, self.num_channels)
            
            
            if self.num_channels==1:
                img = np.expand_dims(img, 2)
            
            # check shape
            if img.shape != self.img_shape:
                raise ValueError("DataGen img.shape Error ", img.shape, self.img_shape)
            
            
            # load mask for y
            mask = load_mask(mask_path, verbose=verbose, dtype=self.mask_type, target_size=self.img_size)
            
            # check shape and values
            if mask.shape != self.img_size:
                print(mask.shape, self.img_size)
                raise ValueError("DataGen error mask.shape mismatch")
            if mask.min()<0 or mask.max()>2:
                raise ValueError("DataGen error mask min max")
            
            # Expand the mask here
            # --------------------
            mask = np.expand_dims(mask, 2)
            
            # For data augmentation assign to temporary list
            # ----------------------------------------------
            if self.data_augment:
                img_list.append(img)
                msk_list.append(mask) 
            # without data augmentation directly assign to volume
            # ---------------------------------------------------
            else:
                # assign image
                x[idx] = img
                # assign mask
                y[idx] = mask
                    
        #print("DataGen x ",x.shape, x.min(), x.max())
        #unique, count = np.unique(y,return_counts=True)
        #print("DataGen y [perc]",y.shape, y.min(), unique, count, y.max())
        
        
        if self.data_augment:
            
            # img_list, mks_list = call augmentation here
            assert self.batch_size == len(img_list)
            assert self.batch_size == len(msk_list)
            
            img_aug_list, msk_aug_list, name_augment   = self.c_transform_affine.augment   (img_list, msk_list)
            img_trf_list, msk_trf_list, name_transform = self.c_transform_nonaffine.augment(img_aug_list, msk_aug_list)
            
            for idx in range(self.batch_size):
                x[idx]=img_aug_list[idx]
                y[idx]=msk_aug_list[idx]
            
            # Augment:
            img_list=[]
            msk_list=[]    
        
        assert y.min()>=0 
        assert y.max()<=2
        
        #return the tuple
        # ---------------    
        return x, y
            
    def save_mask(self, mask, filename, check=True):
        save_mask( mask, filename, check=check)

    
    def on_epoch_end(self):
        pass
        print("*"*80)
        print("%s DataGen OnEpochEnd called "%self.name)
        print("*"*80)
        
    def on_batch_end(self):
        print("*"*80)
        print("%s DataGen OnBatchEnd called "%self.name)
        print("*"*80)
    
'''
---------------------------------------------------------------------------------------------
'''

