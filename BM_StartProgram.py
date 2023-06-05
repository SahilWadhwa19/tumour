'''
Created on 28 Mar 2021

@author: digit
################
'''
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#gedit /home/$USER/.bashrc
#export PYTHONPATH=~/Dropbox/DIGIT_AIDrivers/__Source/__BostonMedical_Source

class BM_StartProgram(object):
    
    def __init__(self, base_file):
        self.base_file = base_file
        self.base_file_name = os.path.basename(base_file)
        self.base_dir = os.path.dirname(base_file)
        try:
            self.python_path = os.environ.get('PYTHONPATH','')
        except BaseException:
            self.python_path=''    
            
        path = self.python_path.split(sep=':')
        
        self.path=[]
        
        # Set environment here
        # --------------------
        cmd_export = "conda activate SAHIL37\n"
        
        cmd_export = cmd_export + "export PYTHONPATH=${PYTHONPATH}"
        for line in path:
            if line.find("_Source")>-1 or line.find("python3.")>-1:
                cmd_export = cmd_export + ":"+line
                print("cmd ",line)
            else:
                #cmd_export = cmd_export + ":"+line
                print("xxx ",line)
                    
        #print(cmd_export)   
        
        cmd = cmd_export + "\n"
        cmd = cmd + "cd %s\npython %s "%(self.base_dir, self.base_file_name)
        print(cmd+"\n")
        
        
        filename="StartUp "+ "_" + os.path.splitext(self.base_file_name)[0] + ".cmd"
        print(filename)
        print(cmd)
        """
        with open(filename, 'w') as f:
            # f.write(cmd+"\n") 
            print("cmd written to ",filename)
         """
        #echo $ PYTHONPATH
        #export PYTHONPATH=${PYTHONPATH}:${HOME}/foo
                    
'''        
            
try:
    print("PYTHONPATH ", os.environ.get['PYTHONPATH'],'')
except BaseException as e:
    pass
print("-"*80)

'''
'''

try:
    path = str(os.environ.get('PYTHONPATH','')).split(sep=':')
except BaseException as e:
    path=""
    
for line in path:
    print(line)
'''    
    
#PYTHONPATH  
#/home/digit/.p2/pool/plugins/org.python.pydev.core_8.2.0.202102211157/pysrc/pydev_sitecustomize:
#/home/digit/Dropbox/DIGIT_AIDrivers/__Source/DigitMedical/UNets/UNet_HDense:
#/home/digit/Dropbox/DIGIT_AIDrivers/__Source/DigitMedical/UNets/DigitUNet:
#/home/digit/Dropbox/DIGIT_AIDrivers/__Source/AIDigit:
#/home/digit/Dropbox/DIGIT_AIDrivers/__Source/DigitMedical/CTScan/CTScanKeras:
#/home/digit/Dropbox/DIGIT_AIDrivers/__Source/Lidar/PointNets/FPointNet:
#/usr/lib/python3.6:/usr/lib/python3.6/lib-dynload:
#/usr/local/lib/python3.6/dist-packages:
#/usr/lib/python3/dist-packages:
#/home/digit/.local/lib/python3.6/site-packages

'''
-------------------------------------------------------------------------------------------------
'''
