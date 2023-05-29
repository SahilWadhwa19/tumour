import os
import io
import numpy as np
import mlflow
import warnings 
# import cv2

from PIL import Image as pil_image
import matplotlib.pyplot as plt
import tensorflow
import tensorflow.keras
from sklearn.preprocessing import label_binarize
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import classification_report as sk_classification_report
from sklearn.metrics import roc_curve, auc
import itertools
from tensorflow import keras
def load_data_paths(data_path):
    
    input_paths = sorted([os.path.join(data_path, file_name) for file_name in os.listdir(data_path)
                         if file_name.endswith(".jpg") or file_name.endswith(".png")])
    return input_paths
  
  

image_path = "/data/img/"
mask_path = "/data/seg"

input_images_paths = load_data_paths(image_path)
input_masks_paths = load_data_paths(mask_path)
print(len(input_images_paths))
print(len(input_masks_paths))
assert len(input_images_paths) == len(input_masks_paths)

def load_img(path, color_mode='grayscale'):
    """Loads an image into PIL format.

    # Arguments
        path: Path to image file.
        color_mode: The desired image format. One of "grayscale", "rgb", "rgba".
            "grayscale" supports 8-bit images and 32-bit signed integer images.
            Default: "rgb".
 
    # Returns
        A PIL Image instance.

    # Raises
        None. No checks performed
    """
    # Read binary
    with open(path, 'rb') as f:
        # Raw binary reading procedure according to Keras
        # ------------------------------------------------
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
            
        return img

# Taken from DigitImageMask
# -------------------------
def load_mask(filename ):
    '''
    # Arguments
        filename: Path to image file.
 
    # Returns
        A PIL Image instance with values [0,1,2.,,,]

    # Raises
        Multiple checks performed. Not tuned for speed.
        
    '''    
    # read mask in grayscale mode [0,255]
    # -----------------------------------
    mask = load_img(filename, color_mode = "grayscale")
    
    # Get value to divide to mask to get [0,1,2]
    # ------------------------------------------
    unique = np.unique(mask)
    
    # Examples: [0,127], [0,127,255], [0,1,2]
    # Number of unique elements > 1
    if len(unique)>1:
        norm_value = unique[1]  
    # Only one single unique mask-value
    # ---------------------------------     
    else:
        # still needs some improvements
        # -----------------------------
        norm_value= 127 
    
    # Crucial here:
    # Divide/Normalize mask-values from [0,12x,255] to [0,1,2]
    # --------------------------------------------------------
    mask = np.asarray(mask, dtype=np.float16)
    mask = np.round(mask/norm_value)
    mask = np.asarray(mask,dtype=np.uint8)
    
    # check and assert result
    # -----------------------
    unique_new = np.unique(mask)
    assert len(unique)==len(unique_new)
    assert mask.min()>=0 and mask.max()<=2 #checks are hardcoded here
    
    return mask
  
sample_image = load_img(input_images_paths[16])
# Converted PIL object into numpy array.
sample_image = np.array(sample_image)
print(sample_image.shape)

sample_mask = load_mask(input_masks_paths[16])
# Converted PIL object into numpy array.
sample_mask = np.array(sample_mask)
print(sample_mask.shape)

assert sample_image.shape == sample_mask.shape

# Sample Image
plt.imshow(sample_image, cmap = "gray")
plt.show()

# Sample Mask
plt.imshow(sample_mask, cmap = "gray")
plt.show()

image_size = sample_image.shape
print(image_size)

num_channels = 1

def get_model(image_size, num_channels):    
    inputs = tensorflow.keras.Input(shape=image_size + (num_channels,))
    
    ### [First half of the network: downsampling inputs] ###
    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    # for filters in [64, 128, 256]:
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
                    previous_block_activation
                )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual
    
    ### [Second half of the network: upsampling inputs] ###
    # for filters in [256, 128, 64, 32]:
    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual
            
        # Add a per-pixel classification layer
    outputs = layers.Conv2D(3, 3, activation="softmax", padding="same")(x)

        # Define the model
    model = tensorflow.keras.Model(inputs, outputs)
        
    return model

model = get_model(image_size, num_channels)
# model.summary()

images_data = []

for path in input_images_paths:
    image_data = np.asarray(load_img(path))
    images_data.append([image_data])
    
print("Images data's length is ", len(images_data))

masks_data = []

for path in input_masks_paths:
    mask_data = np.asarray(load_mask(path))
    masks_data.append([mask_data])
    
print("Masks data's length is ", len(masks_data))

assert len(images_data) == len(masks_data)

print(type(images_data))
images_data = np.asarray(images_data)
print(images_data.shape)

print(len(masks_data))
print(type(masks_data))
masks_data = np.asarray(masks_data)
print(masks_data.shape)

images_data = np.reshape(images_data, (len(images_data), image_size[0], image_size[1], 1))
print(images_data.shape)
print(type(images_data))

masks_data = np.reshape(masks_data, (len(images_data), image_size[0], image_size[1], 1))
print(masks_data.shape)
print(type(masks_data))


# When we need to do the training without ipus
# tensorflow.keras.losses.SparseCategoricalCrossentropy()
# Train the model.

from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import roc_auc_score as sk_roc_auc_score

# Set up MLFlow Experiment
MLFLOW_EXPERIMENT_NAME = os.getenv('DKUBE_PROJECT_NAME')

if MLFLOW_EXPERIMENT_NAME:
    exp = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if not exp:
        print("Creating experiment...")
        mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME)
    mlflow.set_experiment(experiment_name=MLFLOW_EXPERIMENT_NAME)

# Output directory for MLFlow
OUTPUT_MODEL_DIR = os.getcwd()+"/model_mlflow"
os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)

# MLFlow metric logging function
class loggingCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metric("train_loss", logs["loss"], step=epoch)
        mlflow.log_metric ("train_accuracy", logs["accuracy"], step=epoch)
        mlflow.log_metric("val_loss", logs["val_loss"], step=epoch)
        mlflow.log_metric ("val_accuracy", logs["val_accuracy"], step=epoch)
        # output accuracy metric for katib to collect from stdout
        print(f"loss={round(logs['loss'],2)}")
        print(f"val_loss={round(logs['val_loss'],2)}")
        print(f"accuracy={round(logs['accuracy'],2)}")
        print(f"val_accuracy={round(logs['val_accuracy'],2)}")
        actual = np.random.binomial(1,0.9, size = 1000)
        predicted = np.random.binomial(1,0.9, size = 1000)
        from sklearn import metrics
        confusion_matrix = metrics.confusion_matrix(actual, predicted)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [True, False])
        cm_display.plot()
        plt.show()
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
    
                    
    


        
        
model.compile(loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(), 
                  optimizer=tensorflow.keras.optimizers.Adam(),
                  metrics=["accuracy"])
"""
with mlflow.start_run(run_name="tumour") as run:
    model.fit(x=images_data, y=masks_data, epochs=1, verbose=True, validation_split=0.1, callbacks=[loggingCallback()])

    # Exporting model & metrics
    print("Model Save")
    model.save("/model/1")

    print("Artifact Save")
    mlflow.log_artifacts(OUTPUT_MODEL_DIR)
    print("Log Model")
    mlflow.keras.log_model(keras_model=model, artifact_path=None)
    """
with mlflow.start_run(run_name="tumour") as run:
    model.fit(images_data, masks_data, epochs = 1, batch_size = 2, validation_split=0.1, callbacks=[loggingCallback()])

    print("Data type is ", images_data.dtype)
    images_data = np.float32(images_data)

    y_pred = model.predict(images_data)
    y_pred = np.argmax(y_pred, axis = -1)
    y_test = masks_data
    y_test = np.reshape(masks_data, (len(images_data), image_size[0], image_size[1]))
    print("ConfMat y_test ",y_test.shape, y_test.dtype)
    print("ConfMat y_pred ",y_pred.shape, y_pred.dtype)
    assert y_test.shape == y_pred.shape, "Mismatch y_test.shape == y_pred.shape"

    # ravel data to 1dim arrays
    y_test = y_test.ravel()
    y_pred = y_pred.ravel()
    print("y_test, y_pred ", np.asarray(y_test).shape, np.asarray(y_pred).shape)
    assert np.asarray(y_test).shape == np.asarray(y_pred).shape, "Mismatch y_test.shape vs y_pred.shape"
    assert len(y_test.shape)==1, "Mismatch len(y_test.shape)==1"
    assert len(y_pred.shape)==1, "Mismatch len(y_pred.shape)==1"
    cnf_matrix = (sk_confusion_matrix(y_test, y_pred))
    print(cnf_matrix)        

    # Exporting model & metrics
    print("Model Save")
    model.save("/model/1")
    print("TrainGen: Training completed")
