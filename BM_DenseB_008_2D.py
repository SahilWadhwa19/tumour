'''
Created on 16 Jan 2023

@author: digit
'''


'''
Paper: DenseUNets with feedback non-local attention for the segmentation of 
       specular microscopy images of the corneal endothelium with Fuchs dystrophy
https://github.com/alexandrosstergiou/keras-DepthwiseConv3D/blob/master/DepthwiseConv3D.py
https://www.tensorflow.org/tutorials/video/video_classification

'''
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import add, concatenate
from tensorflow.keras.layers import Input, Dropout, Activation, BatchNormalization



from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Concatenate

# Own imports for Network
# -----------------------
from BM_DenseUnet_Tools import get_layers_list, rename_layers, get_network_IDs

# Own imports 
# -----------
from DigitKerasModel_dkube import DigitKerasModel
#from DigitKerasMemory import DigitKerasMemory
#from DigitKerasPlot import display_model

# generalization 2D/3D
# --------------------
concat_axis = 3 #DIGIT: channels_last per definition hardcoded in tf >1.x
PRE='2D_'


class BM_DenseB_008_2D(DigitKerasModel):
    
    def __init__(self, 
                 #training
                 batch_size=32, 
                 # input
                 input_shape = (512, 512, 1),
                 input_size=512, 
                 num_cols=1,#8, 
                 num_channels=1, 
                 num_classes=3, #constant
                 # model architecture
                 # ------------------
                 network_ID=169,
                 # network details
                 # ---------------
                 reduction_rate = 0.5,
                 growth_rate=8,
                 use_trans=True,
                 filters=[24, 48, 128, 128],
                 # paths
                 modelname=None,
                 dirname=None):
        
        # training
        self.batch_size = batch_size
        # input
        self.input_shape = input_shape
        self.input_size = input_shape[0]
        self.num_cols = num_cols
        self.num_channels = num_channels
        self.num_classes = num_classes #constant
        
        # model architecture
        self.network_ID=network_ID
        
        print("-"*80)
        print("Network ID ",network_ID)
        print("-"*80)

        # Get layers list and name for ID
        # -------------------------------
        self.layers_list, self.network_name = get_layers_list(network_ID)
        
        self.reduction_rate = reduction_rate
        self.growth_rate = growth_rate
        self.use_trans = use_trans
        self.filters=filters
        # paths and name
        # --------------
        # paths and name of model
        self.dirname = dirname or os.path.join(BASE_DIR,"_architecture")
        
        
        if modelname is None:
            self.modelname = type(self).__name__ + "_ID%d_"%network_ID +"gr%d"%self.growth_rate+"_tr%d"%self.use_trans
            modelname=self.modelname
        else:
            self.modelname = modelname
            
        # init base class
        # ---------------
        DigitKerasModel.__init__(self, dirname=self.dirname, modelname=self.modelname)


    def get_model(self, 
                  img_input=None, 
                  # architecture
                  # ------------
                  network_ID=None,
                  layers_list = (4, 8, 12, 16, 20), 
                  input_shape = (512, 512, 1), 
                  num_classes = 3,
                  # parameter
                  # ---------
                  growth_rate = 8,#5 
                  drop_rate = 0.2,
                  use_trans=True):
        
        network_ID = network_ID or self.network_ID
    
        # DIGIT
        # -----
        if img_input is None:
            # DIGIT: 2D/3D Hybrid requires known batch_size
            # Therefore Input(batch_shape=(xxx)
            img_input = Input(batch_shape=(self.batch_size, 
                                           self.input_size, self.input_size, 
                                           self.num_channels), 
                                           name=PRE+'volumetric_input')

    
        layers_list=self.layers_list
        self.model = self.dense_UNet(img_input=img_input,
                                # architecture 
                                network_ID=network_ID, 
                                layers_list=layers_list, 
                                input_shape=self.input_shape, 
                                # details
                                growth_rate=self.growth_rate, 
                                drop_rate=0.2, 
                                use_trans=self.use_trans,
                                filters=self.filters,
                                num_classes=3)
        
        return self.model
                  
    def dense_UNet(self, img_input=None, 
                   # general
                   # -------
                   input_shape=(512, 512, 1), 
                   num_classes=3,
                   # architecture
                   # ------------
                   network_ID=99, 
                   layers_list=(4, 8, 12, 16, 20), 
                   # network details
                   # ---------------
                   growth_rate=8, 
                   drop_rate=0.2,
                   use_trans=False,
                   filters=None):
        
        
            """ Instantiates the DenseUNet architecture. Optionally loads weights.
            # Arguments
                blocks:       numbers of building blocks for the four dense layers.
                input_shape:  tuple, (H,W,C)    
                growth_rate:  int, the number of feature maps in each convolution within 
                              the dense blocks.      
                drop_rate:    float, the dropout rate (0-1), or None if not used.                       
                classes:      int, number of classes to classify images into, only to  
                              be specified if no `weights` argument is specified
            # Returns
                A TensorFlow-Keras model instance.
            # Raises
                ValueError: in case of invalid argument for `weights`,
                            or invalid input shape.
            """
        
            #BM_DenseB_003_2D
            #----------------
            '''
            implement for loops for downsampling
            implement lists for skips and filters
            implement layers
            '''
        
            #BM_DenseB_004_2D
            #----------------
            '''
            implement network_ID ans layers_list
            '''
            filters_169= [24, 48, 128, 128]
    
            if filters is None:
                filters = [int(x*growth_rate*0.50) for x in layers_list]
            else:
                assert len(filters)==len(layers_list), "Mismatch filters vs layers_list"    
                print("filters "*10)
            print("filters ",filters, " growth_rate ",growth_rate)
            
            # Input should be set 
            # -------------------
            if img_input is None:   
                img_input = Input(shape=input_shape)
                
            
            print("*"*100)
            print("Model architecture: ")
            print("\t Model name    \t",self.modelname)
            
            print("\t Input.shape   \t",img_input.shape)
            print("\t layers_list   \t",layers_list)
            print("\t filters       \t",filters)
            print("\t filters_169   \t",filters_169)
            
            print("\t Network_ID    \t",network_ID)
            print("-"*100)
            print("\t growth_rate   \t",growth_rate)        
            print("\t dropout_rate  \t",drop_rate)
            print("\t use_trans     \t",use_trans)
            
            print("-"*100)
            print("")
            
            if not use_trans:
                print("-"*100)
                print("No transition used "*4)
                print("-"*100)
            
            '''
            print("\t nb_filter     \t",nb_filter)
            print("\t compression   \t",compression_rate) 
            print("\t reduction     \t",reduction_rate)
            print("\t allow_growth  \t",allow_growth)
            
            print("")
            
            print("*"*100)
            print("\n")
            ''' 
            # 
            skips=[]
            nb_dense_blocks = len(layers_list)
            
            for idx in range(0, nb_dense_blocks):
                print("Downsampling ", idx)
                
                name_conv='conv%d0'%idx
                name_tran='tran%d0'%idx
                name_down='down%d0'%idx
                
                if idx==0:
                    # DIGIT: Checked that dense goes on layers_list[idx] while transition goes on filters[idx]
                    x_dense = dense_block_initial(img_input, layers_list[idx], growth_rate, drop_rate=None, name=name_conv)#DIGIT: No dropout
                    x_trans = transition_block(x_dense,   filters[idx], name=name_tran)
                    if use_trans: skips.append(x_trans) # DIGIT: confirmed that downsampling works on x_dense and not x_trans
                    x_down = downsampling_block(x_dense, filters[idx], name=name_down)
                else:
                    x_dense = dense_block(x_down, layers_list[idx], growth_rate, drop_rate=drop_rate, name=name_conv)
                    
                    # last layer has no trans:
                    # ------------------------
                    if idx==nb_dense_blocks-1:
                        print("skip transition block ",idx)
                        x_down=x_dense
                        # DIGIT: The bottom layer is result of UP-sampling
                        # ------------------------------------------------
                        bottom = upsampling_block(x_down, filters[idx], growth_rate, name='upsa40')
                        
                        # The last dense block at the end of the contracting path is fed into 4 convolutional layers
                        # with dilation 2, 4, 8, and 16. 
                        # Once the blocks go through the dilated convolutions, they
                        # are concatenated to gain wider spatial perspective at the end of the contracting path of the
                        # dilated dense UNet.
                        
                    else:
                        x_trans = transition_block( x_dense, out_channels=filters[idx], name=name_tran)
                        if use_trans: skips.append(x_trans)
                        # DIGIT: layers in dense-block are result of DOWN-sampling
                        # --------------------------------------------------------
                        x_down = downsampling_block(x_dense, out_channels=filters[idx], name=name_down)    
            
            print("filters ",filters)            
        
            for idx, skip in enumerate(skips):
                print("skips.shape ", idx, skip.shape, np.product(skip.shape)/10e6)
                '''
                skips.shape  0 (4, 256, 256, 96)
                skips.shape  1 (4, 128, 128, 384)
                skips.shape  2 (4, 64, 64, 768)
                skips.shape  3 (4, 32, 32, 2112)
                '''
            
            for idx, filter_ in enumerate(filters):
                print("block %d: filter %d"%(idx, filter_))
                # DIGIT: filters a bit redundant since filter[idx] == skips[idx].shape[-1]
                # assert filter_ == skips[idx].shape[-1],"Mismatch filter and skips"
        
        
            print("")
            print("Upsampling")
            print("-"*200)
            
            cnt=0
            # DIGIT: bottom layer is upsampled
            # ---------------------------------
            for idx, filter in reversed(list(enumerate(filters))):
                if idx==0:
                    break
                if idx==len(filters)-1:
                    print("upsampling idx skipped ",idx)
                    continue
                cnt+=1
                print("Upsampling reverse ", idx, filter, len(filters))
                name_conc ='conc%d%d'%(idx,cnt)
                name_dense='conv%d%d'%(idx,cnt)
                name_down ='upsa%d%d'%(idx,cnt)
                if idx == len(filters)-2:
                    print("concat bottom ",bottom.shape)
                    x_up = bottom
                if use_trans:
                    x_conc = Concatenate(name=name_conc)([skips[idx],x_up])
                    print("conc ", idx, skips[idx].shape, x_up.shape, " --> ",x_conc.shape)
                    print("skips  ",idx, skips[idx])
                    print("x_up   ",idx, x_up)
                    print("x_conc ",idx, x_conc)
                    #print(x_conc.name, x_conc.get)
                    print("-"*80)
                    
                else:
                    x_conc = x_up
                # DIGIT: First dense then upsampling
                # ----------------------------------        
                x_dense = dense_block(x_conc, layers_list[idx], growth_rate, drop_rate=drop_rate, name=name_dense)
                x_up = upsampling_block(x_dense, filters[idx],  growth_rate, name=name_down)
            
            # final layer
            # -----------
            if use_trans:
                final = Concatenate(name='conc_final')([skips[0],x_up])
            else:
                final = x_up    
            # feature
            # -------
            feature = dense_block(final, layers_list[0], growth_rate, drop_rate=None, name='conv_fin')
            
            # classifier
            # ----------
            classifier = transition_block(feature, num_classes, name='conv05_feat', activation='relu')
            classifier = Activation('softmax', name='conv05_class')(classifier)
        
            print("Check output ",classifier.shape, classifier.shape[:3], img_input.shape[:3])
            assert classifier.shape[:3]==img_input.shape[:3]
            assert classifier.shape[-1]== num_classes
            assert classifier.shape[0]== self.batch_size*self.num_cols
            print("Model checked...")
        
            model = Model(inputs=img_input, outputs=classifier, name='denseUnet_004')
            return model

def dense_block_initial(x, blocks, growth_rate, name, drop_rate=None, activation='relu'):
    """The first dense block, without Conv1x1 (feature reduction).
    # Arguments
        x:           input tensor.
        blocks:      integer, the number of building blocks.
        growth_rate: float, growth rate at dense layers (output maps).
        name:        string, block label.
        drop_rate:   float, the dropout rate (0-1), or None if not used.
        activation:  string, the type of activation
    """
    for idx in range(blocks):
        # Default: strides=(1, 1)
        x1 = Conv2D(growth_rate, 3, padding='same', use_bias=False, activation=None, name=name + '_block' + str(idx+1) + '_conv')(x)
        x1 = BatchNormalization(epsilon=1.001e-5, renorm=True, name=name + '_block' + str(idx+1) + '_bn')(x1)  
        x1 = Activation(activation, name=name + '_block' + str(idx+1) + '_actv')(x1)
        # DIGIT: Not sure wether inital should have dropout
        if drop_rate is not None:
            x1 = Dropout(rate=drop_rate, name=name + str(idx+1) + '_drop')(x1)
        x = Concatenate(name=name + '_block' + str(idx+1) + '_concat')([x, x1])
    return x


def dense_block(x, blocks, growth_rate, name, drop_rate=None, activation='relu'):
    """A dense block. It constitutes several conv_blocks
    # Arguments
        x:           input tensor.
        blocks:      integer, the number of building blocks.
        growth_rate: float, growth rate at dense layers (output maps).
        name:        string, block label.
        drop_rate:   float, the dropout rate (0-1), or None if not used.
        activation:  string, the type of activation
    """
    for idx in range(blocks):
        # Default: strides=(1, 1)
        x = conv_block(x, growth_rate, drop_rate=drop_rate, activation=activation, name=name + '_block' + str(idx+1))
    
    #DIGIT: No dropout
    
    return x


def conv_block(x, growth_rate, name, drop_rate=None, activation='relu'):
    """ A building block for a dense block.
    # Arguments
        x:           input tensor.
        growth_rate: float, growth rate at dense layers.
        name:        string, block label.
        drop_rate:   float, the dropout rate (0-1), or None if not used.
        activation:  string, the type of activation
    # Returns
        Output tensor for the block.
    """
    # Default: strides=(1, 1)
    x1 = Conv2D(4 * growth_rate, 1, use_bias=False, activation=None, name=name + '_1_conv')(x)
    x1 = BatchNormalization(epsilon=1.001e-5, renorm=True, name=name + '_1_bn')(x1) #
    x1 = Activation(activation, name=name + '_1_actv')(x1)
    
    # Default: strides=(1, 1)
    x1 = Conv2D(growth_rate, 3, padding='same', use_bias=False, activation=None, name=name + '_2_conv')(x1)
    x1 = BatchNormalization(epsilon=1.001e-5, renorm=True, name=name + '_2_bn')(x1) 
    x1 = Activation(activation, name=name + '_2_actv')(x1)
    if drop_rate is not None:
        x1 = Dropout(rate=drop_rate, name=name + '_drop')(x1)
    
    # DIGIT: here is concatenate
    x = Concatenate(name=name + '_concat')([x, x1])
    return x
  

def transition_block(x, out_channels, name, activation='relu'):
    """ A transition block, at the end of the dense block, without including 
    the downsampling.
    # Arguments
        x:            input tensor.
        out_channels: int, the number of feature maps in the convolution.
        name:         string, block label.
        activation:   string, the type of activation
    # Returns
        output tensor for the block.
    """
    # Default: strides=(1, 1)
    x = Conv2D(out_channels, 1, activation=None, use_bias=False, name=name + '_conv')(x)
    x = BatchNormalization(epsilon=1.001e-5, renorm=True, name=name + '_bn')(x) #
    x = Activation(activation, name=name + '_actv')(x)
    return x


def downsampling_block(x, out_channels, name, activation='relu'):
    """ An upsampling block with tranpose convolutions.
    # Arguments
        x:            input tensor.
        growth_rate:  float, growth rate at the first convolution layer.
        out_channels: int, the number of feature maps in the convolution.
        name:         string, block label.
        activation:   string, the type of activation
    # Returns
        output tensor for the block.
    """
    x = Conv2D(out_channels, 2, strides=2, activation=None, padding='same', use_bias=False, name=name + '_conv')(x)
    x = BatchNormalization(epsilon=1.001e-5, renorm=True, name=name + '_bn')(x) 
    x = Activation(activation, name=name + '_actv')(x)
    return x

      
def upsampling_block(x, out_channels, growth_rate, name, activation='relu'):
    """ An upsampling block with tranpose convolutions.
    # Arguments
        x:            input tensor.
        growth_rate:  float, growth rate at the first convolution layer.
        out_channels: int, the number of feature maps in the convolution.
        name:         string, block label.
        activation:   string, the type of activation
    # Returns
        output tensor for the block.
    """
    x = Conv2DTranspose(out_channels, 2, strides=2, activation=None, padding='same', use_bias=False, name=name + '_conv')(x)
    x = BatchNormalization(epsilon=1.001e-5, renorm=True, name=name + '_bn')(x) #
    x = Activation(activation, name=name + '_actv')(x)
    return x



'''
----------------------------------------------------------------------------------------------------------
'''
if __name__ == '__main__': 

    batch_size=8
    input_cols=1
    network_IDs=[99, 121, 161, 169, 201]
    network_IDs=[169]
    use_trans=True
    
    filters= [24, 48, 128, 128]#Total params: 2,032,069
    filters= [24, 48,  96,  96]#Total params: 1,714,245
    filters= [24, 48,  48,  48]#Total params: 1,283,589
    filters= [ 2,  4,   8,  16]#Total params:   837,953
    
    # Original filter
    # ---------------
    filters= [24, 48, 128, 128]
    for network_ID in network_IDs:
        
    
        c_model = BM_DenseB_008_2D(batch_size=batch_size, 
                                   network_ID=network_ID,
                                   use_trans=use_trans,
                                   filters=filters)  
        model = c_model.get_model()
        #model.summary()
        c_model.save_summary()

    print("Program terminated")  
        
#feedback-non-local-attention-fNLA/denseunet.py at master · jpviguerasguillen/feedback-non-local-attention-fNLA · GitHub
'''
------------------------------------------------------------------------------------------------------------
'''
