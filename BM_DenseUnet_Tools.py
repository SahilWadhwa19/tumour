'''
Created on Jan 4, 2023

@author: digit
'''

def get_layers_list(ID):
    # Define and check model
    # ----------------------
    if ID== 121:
        nb_layers = [6, 12, 24, 16]
        name='densenet121'
    elif ID==169:
        nb_layers = [6, 12, 32, 32]
        name='densenet169'
        
        name="UH_DenseUnet"
    elif ID==201:
        nb_layers = [6, 12, 48, 32]
        name='densenet201'
        
    elif ID==0:
        nb_layers = [3,  4, 12,  8] 
        name="UH_Main3D_original"
        
    elif ID==161: 
        nb_layers = [6, 12, 36, 24] # For DenseNet-161
        name = "densenet161"
    else:
        name='densenet_undefined'
        print("layers_list ", nb_layers)
        raise ValueError("Invalid layers list ",nb_layers)
    return nb_layers, name    
    
def get_network_IDs():
    return [0, 121, 161, 169]

def define_model(self,nb_layers):
    # Define and check model
    # ----------------------
    if nb_layers ==   [6, 12, 24, 16]:
        name='densenet121'
        ID=121
    elif nb_layers == [6, 12, 32, 32]:
        name='densenet169'
        ID=169
        name="UH_DenseUnet"
    elif nb_layers == [6, 12, 48, 32]:
        name='densenet201'
        ID=201
    elif nb_layers == [3,  4, 12,  8]: 
        name="UH_Main3D_original"
        ID=0
    elif nb_layers == [6, 12, 36, 24]: # For DenseNet-161
        name = "densenet161"
        ID=161
    else:
        name='densenet_undefined'
        print("layers_list ", nb_layers)
        raise ValueError("Invalid layers list ",nb_layers)
    return name, ID    

def rename_layers(model, PRE='2D_'):
    '''
    Helper to rename layers with given prefix
    '''
    for layer in model.layers:
        name = str(layer.name)
        #print("",name)
        if name.find(PRE)<0: #DIGIT: 
            layer._name = PRE+name
        name = str(layer.name)
        assert name.find(PRE)>=0
    return model  
'''
--------------------------------------------------------------------
'''    