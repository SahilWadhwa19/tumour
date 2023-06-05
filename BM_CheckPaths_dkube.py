'''
Created on Mar 10, 2022

@author: digit
'''
import os
import numpy as np

def check_paths_synch(img_paths, mask_paths)->bool:
    
    if len(img_paths) != len(mask_paths):
        print(" check_paths_synch Size mismatch img & mask files ", len(img_paths),len(mask_paths))
        return False
    ret = True
    for idx in range(len(img_paths)):
        img_path  = os.path.basename(img_paths [idx])
        mask_path = os.path.basename(mask_paths[idx])
        
        img_path = img_path[:-4]
        img_path = img_path[3:]
        
        mask_path = mask_path[:-4]
        mask_path = mask_path[3:]
        
        if img_path != mask_path:
            print ("check_paths_synch Load error for ",img_path, mask_path)
            ret = False
    return ret    

'''
----------------------------------------------------------------------
'''


        
        
