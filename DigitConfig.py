"""
Created on Fri Dec 14 15:43:23 2018

@author: DIGIT
"""
import os
import sys
import json
import socket

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ROOT_DIR = os.path.dirname(BASE_DIR)
print("BASE_DIR (AIConfig)", BASE_DIR)
print("ROOT_DIR (AIConfig)", ROOT_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)


def Config(filename = '_BostonMedical', use_local=True):
    # get extension according to computer name
    # ----------------------------------------
    extension = getConfigExtension(filename, use_local=use_local)
    conf_file = os.path.join(BASE_DIR, extension) 
    
    if not os.path.isfile(conf_file):
        # check in root directory
        # -----------------------
        conf_dir = os.path.join(ROOT_DIR, filename) 
        if not os.path.isdir(conf_dir):
            raise ValueError('Conf Dir %s does not exist'% conf_dir)
        conf_file = os.path.join(conf_dir, extension) 
    
    print("Conf file ",conf_file)
    if not os.path.isfile(conf_file):
        raise ValueError("Config file %s does not exist"% conf_file)
    
    # open and read json
    # ------------------
    with open(conf_file, "r") as fp:
        conf = json.load(fp)
    return conf

def getConfigExtension(filename, use_local=True):
 
    if use_local:
        computername= socket.gethostname()
    else:
        computername=''    
    
    # current convention for config files 
    # with/without computername
    # -------------------------
    filename = filename + "_Conf_" + computername + ".json"
    return filename

if __name__ == "__main__":
    
    print("computername ", socket.gethostname())
    
    conf = Config('_BostonMedical')
    print("LIVER_PATH [%s]"%str(conf["LIVER_PATH"]))
    
    data_path = conf["LIVER_PATH"]
    assert os.path.isdir(data_path), "Data_path does not exist"
    print("Program terminated")
    
    
'''
---------------------------------
'''    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
