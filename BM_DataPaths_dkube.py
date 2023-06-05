'''
Created on 31.01.2021

@author: DIGIT
'''
import os

# Own imports
# -----------
# None
# ----


class BM_DataPaths(object):
    def __init__(self, data_path):
        
        self.data_path = data_path
        
        if not os.path.isdir(self.data_path):
            raise ValueError("Data dir does not exist ",self.data_path)
    
    def load_file_list(self, dirname, filename, verbose=False):
        files=[]
        if not os.path.isfile(filename):
            raise ValueError("load_file_list file does not exist %s "%filename)
        if not os.path.isdir(dirname):
            raise ValueError("load_file_list dir does not exist %s "%dirname)
        
        with open(filename, "r") as fp:
            lines = [line.rstrip('\n') for line in fp]
            for line in lines:
                if len(line)>0:
                    file = os.path.join(dirname, line)
                    files.append(file)
                    #print("file.append [%s] "%file)
                    if not os.path.isfile(file):
                        raise ValueError("Data file %s does not exist "%file)
        
        if verbose:
            print("%d files loaded from seg_file %s"%(len(files), filename))
        return files            
        
    def load_data_paths(self):
        # get whole dir for model-pipeline
        # --------------------------------
        self.input_paths = sorted([os.path.join(self.data_path, fname) for fname in os.listdir(self.data_path) 
                if fname.endswith(".jpg") or fname.endswith(".png")])
        print("load_data_paths: Number of available items:", len(self.input_paths))
        return self.input_paths
                   
    def select_data_paths_for_ID(self, liver_ID, slice_ID=None, file_start="seg"):
        # get whole dir for model-pipeline
        # --------------------------------
        if slice_ID is None:
            file_search = "%s%06d"%(file_start, liver_ID)
        else:    
            file_search = "%s%06d_%04d"%(file_start, liver_ID, slice_ID)
        
        input_paths = sorted([os.path.join(self.data_path, fname) for fname in os.listdir(self.data_path) 
                if fname.find(file_search)>=0 and (fname.endswith(".jpg") or fname.endswith(".png"))])
        print("Number of available items:", len(input_paths))
        return input_paths



'''
-----------------------------------------------------
'''    
if __name__ == '__main__':  
    pass
'''
-----------------------------------------------------
'''    
    

    
    
