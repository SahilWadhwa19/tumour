'''
Created on 14 Jan 2022

@author: digit
'''
#import io
import numpy as np
#import warnings
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import load_img as tf_load_img 

def load_img(path, grayscale=False, color_mode="grayscale", target_size=None,
             interpolation='nearest', verbose=True):

    img= tf_load_img(
                path=path,
                grayscale=grayscale,
                color_mode=color_mode,
                target_size=target_size,
                #interpolation='nearest',
                #keep_aspect_ratio=False)
                interpolation='nearest')
                #keep_aspect_ratio=False)
    img_arr = tf.keras.utils.img_to_array(img)#(512,512,1)
    
    #input_arr = np.array([input_arr])[0]#(512,512,1)
    img_arr = np.squeeze(img_arr)
    #if verbose: print("input_array ",input_arr.shape)#(512,512)
    return img_arr

'''


#from PIL import ImageEnhance
from PIL import Image as pil_image
_PIL_INTERPOLATION_METHODS = {
        'nearest': pil_image.NEAREST,
        'bilinear': pil_image.BILINEAR,
        'bicubic': pil_image.BICUBIC,
    }

def load_img(path, grayscale=False, color_mode='rgb', target_size=None,
             interpolation='nearest'):
    """Loads an image into PIL format.

    # Arguments
        path: Path to image file.
        grayscale: DEPRECATED use `color_mode="grayscale"`.
        color_mode: The desired image format. One of "grayscale", "rgb", "rgba".
            "grayscale" supports 8-bit images and 32-bit signed integer images.
            Default: "rgb".
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", and "bicubic".
            If PIL version 1.1.3 or newer is installed, "lanczos" is also
            supported. If PIL version 3.4.0 or newer is installed, "box" and
            "hamming" are also supported.
            Default: "nearest".

    # Returns
        A PIL Image instance.

    # Raises
        ImportError: if PIL is not available.
        ValueError: if interpolation method is not supported.
    """
    if grayscale is True:
        warnings.warn('grayscale is deprecated. Please use '
                      'color_mode = "grayscale"')
        color_mode = 'grayscale'
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `load_img` requires PIL.')
    with open(path, 'rb') as f:
        img = pil_image.open(io.BytesIO(f.read()))
        if color_mode == 'grayscale':
            # if image is not already an 8-bit, 16-bit or 32-bit grayscale image
            # convert it to an 8-bit grayscale image.
            if img.mode not in ('L', 'I;16', 'I'):
                img = img.convert('L')
        elif color_mode == 'rgba':
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
        elif color_mode == 'rgb':
            if img.mode != 'RGB':
                img = img.convert('RGB')
        else:
            raise ValueError('color_mode must be "grayscale", "rgb", or "rgba"')
        if target_size is not None:
            width_height_tuple = (target_size[1], target_size[0])
            if img.size != width_height_tuple:
                if interpolation not in _PIL_INTERPOLATION_METHODS:
                    raise ValueError(
                        'Invalid interpolation method {} specified. Supported '
                        'methods are {}'.format(
                            interpolation,
                            ", ".join(_PIL_INTERPOLATION_METHODS.keys())))
                resample = _PIL_INTERPOLATION_METHODS[interpolation]
                img = img.resize(width_height_tuple, resample)
    img = np.asarray(img)
    ret = np.copy(img)
    return ret
    

def load_mask(filename, color_mode = "grayscale",dtype=np.uint8):
    mask = load_img(filename, color_mode = color_mode)
    unique = np.unique(mask)
    value = unique[1] if len(unique)>1 else 127
    
    mask = np.asarray(mask, dtype=np.float16)
    mask = np.round(mask/value)
    mask = np.asarray(mask,dtype=dtype)
    
    #print("file ",file, mask.min(),np.unique(mask), mask.max())
    if not check_mask_values(mask):
        print("file ",filename, mask.dtype, mask.min(),np.unique(mask), mask.max())
        raise ValueError("check_mask_dir: Check failed")
    return mask
'''

# Taken from DigitImageMask
# -------------------------
def load_mask(filename, color_mode = "grayscale", verbose=False, 
              dtype=np.int16, target_size=None):#DigitDebug
    try:
        mask = load_img(filename, color_mode = color_mode, target_size=target_size)
        #print("mask[0,0] ",mask[0,0], mask.dtype)
        mask = np.asarray(mask)
        mask[0,0]=0#255 #DigitDebug
        unique, count = np.unique(mask,return_counts=True)
        mask_max = mask.max()
        if len(unique)>3:
            print("load_mask error unique, count",unique, count)
            raise ValueError("load mask error wrong number of classes ",len(unique))
        
        if len(unique)==3:
            value = float(unique[1])-1
        else:     
            value = float(mask_max)-1
        
        #value=float(value)#-1    
        if verbose: print("mask unique ",mask.min(), unique, mask.max(), "-->", value)
        
        mask = np.asarray(mask, dtype=np.float16)
        mask = np.fix(mask/value)
        assert mask.max() >=   0.0
        assert mask.min() <  256.0
        mask = np.asarray(mask,dtype=dtype)
        
        unique_new, count_new = np.unique(mask,return_counts=True)
        if verbose: print("mask unique_new ",unique_new, count_new)
        
        assert len(unique)==len(unique_new)
        assert len(count) ==len(count_new)
        
        if verbose: print("check mask ",count, count_new)
        if verbose: print("*"*100)
        if not np.array_equal(count, count_new):
            print("load_mask error ", filename)
            print("mask  in ", unique, count)
            print("mask  new", unique_new, count_new)
            raise ValueError("load_mask error")
        
        assert mask.min()>=0 and mask.max()<=2

    except BaseException as e:
        raise ValueError("load mask error ",str(e))    
    
    return mask

def check_mask_values(mask, target_unique=np.asarray([0,1,2])): #See DigitImageMask
    
    unique = np.unique(mask)
    for u in unique:
        if not np.isin(u,target_unique):
            print("Value is not in target_unique ", u, target_unique)
            return False
    return True 



def save_img(arr, filename, colour=False, verbose=False):
    try:
        assert arr.max()<256
        assert arr.min()>=0
        
        
        
        filename=filename.replace(".png",".jpg")
        if not filename.endswith(".jpg"):
            filename=filename+".jpg"
        if verbose: print("sage_img save to ",filename)    
        if colour:
            plt.imsave(filename, arr)
        else:    
            plt.imsave(filename, arr, cmap = plt.cm.gray)
        
    except BaseException as e:
        print("save_img error ",str(e))        
    
def save_mask( mask, filename, check=True): #See UH_Base and DigitImageMask
    if not filename.endswith(".png"):
        filename=filename+".png"
    try:    
        if check: assert check_mask_values(mask)
                    
        mask = mask*127
        mask[0,0]=255
        if check: assert mask.min()>=0
        if check: assert mask.max()<256
        plt.imsave(filename, mask, cmap = plt.cm.gray)
    except BaseException as e:
        print("save_mask error ",str(e))
            
'''
-----------------------------------------------------
'''    
if __name__ == '__main__':  
    pass
    arr = np.asarray([126/127, 127/127, 128/127, 1.0, 1.1,255/127])
    print("np.round ", np.asarray((np.fix(arr)), dtype=np.uint8))
    exit()
    
    
    a =  [12468437,  3975170,   333602]
    b =  [12468437,  3975170,   333609]
    ret= np.array_equal(a, b)
    print("arrays equal ", ret,"\n",a, "\n",b)
    


'''
------------------------------------------------------------------
'''
    