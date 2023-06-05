'''
Created on 13 May 2021

@author: digit
'''
import os
import numpy as np
import re
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class DigitFileDir(object):
    
    def __init__(self, dir_path="", item_name=None, item_extension=None, item_sort=True):
        
        self.dir_path = dir_path
        self.item_name = item_name
        self.item_extension = item_extension
        self.item_sort = item_sort
    
        self.filelist = []
        
    def get_file_list(self, dir_path=None, item_name=None, item_extension=None, item_sort=True):
        
        if dir_path is None:
            dir_path = self.dir_path
        if not os.path.isdir(dir_path):
            raise ValueError("data_path % s does not exist"% dir_path)
        
        # get whole dir
        # -------------
        filelist = os.listdir(dir_path)
        
        # reduce to item_name
        # -------------------
        if item_name is not None:
            filelist = [item for item in filelist if item_name in item]
        
        # reduce to extension(s)
        # ----------------------
        if item_extension is not None:
            #  convert to list if extension is a string
            # -----------------------------------------
            if isinstance(item_extension, str):
                item_extension = [item_extension]
            filelist = [item for item in filelist if any(item.endswith(ext) for ext in item_extension)]
        
        # sort filelist
        # -------------
        if len(filelist)>0:
            if item_sort: 
                try:           
                    filelist.sort(key=lambda f: int(re.sub('\D', '', f)))
                except BaseException as e:
                    pass
            else:
                try:           
                    filelist.sort()
                except BaseException as e:
                    pass
                        
        self.filelist=filelist
        return filelist
    
    def print_count(self):
        
        print("Dir %s has %d items for item [%s]"%(self.dir_path, len(self.filelist), self.item_name))
    
    def print_list(self):
        
        if len(self.filelist) ==0:
            self.get_file_list()
        for f in self.filelist:
            print("file %s"% f)
            
            
    def get_dirname(self, dirname):
        try:
            dirname = os.path.dirname(dirname) 
            return os.path.split(dirname)[-1]
           
        except BaseException:
            return ""             
'''
------------------------------------------------------------------------------
'''
if __name__ == '__main__':
    dir_path = '/home/digit/Downloads'
    item_name = None
    c_dir = DigitFileDir(dir_path)
    
    #===========================================================================
    # filelist = c_dir.get_file_list(dir_path)
    # print("Dir %s has %d items for item [%s]"%(dir_path, len(filelist), item_name))
    # for f in filelist:
    #     
    #     print("file %s"% f)
    #===========================================================================
    
    item_name=None#'h5'
    item_extension=['.h5','.pdf']
    item_sort=True
    filelist = c_dir.get_file_list(dir_path, item_name=item_name, item_extension=item_extension, item_sort=item_sort)
    c_dir.print_count()
    c_dir.print_list()
        
    print("\n Program terminated")    
'''
------------------------------------------------------------------------------
'''

