# Copyright 2020 MONAI Consortium 
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 
#     http://www.apache.org/licenses/LICENSE-2.0 
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 
 
import glob 
import logging 
import os 
from pathlib import Path 
import shutil 
import sys 
import tempfile 
 
import nibabel as nib 
import numpy as np 
from monai.config import print_config 
from monai.data import ArrayDataset, create_test_image_3d, decollate_batch 
from monai.handlers import ( 
    MeanDice, 
    MLFlowHandler, 
    StatsHandler, 
    TensorBoardImageHandler, 
    TensorBoardStatsHandler, 
) 
from monai.losses import DiceLoss 
from monai.networks.nets import UNet 
from monai.transforms import ( 
    Activations, 
    AddChannel, 
    AsDiscrete, 
    Compose, 
    LoadImage, 
    RandSpatialCrop, 
    Resize, 
    ScaleIntensity, 
    EnsureType, 
) 
from monai.inferers import sliding_window_inference 
from monai.utils import first 
from monai.visualize import plot_2d_or_3d_image 
 
import ignite 
import torch 
from torch.utils.tensorboard import SummaryWriter 
 
print_config()
