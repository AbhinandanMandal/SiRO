
"""
===============================================================================
Code for State-invariant Object Representation (SIOR) using 
Pose-invariant Attention Network to learn category and object embeddings in the
same space by training jointly using L-Softmax and Pose-invariant losses and
curriculum learning for better results
===============================================================================
"""

"""
In PiRO code (https://github.com/sarkar-rohan/PiRO), 
author used pose invariant object loss and large margin softmax loss for 
single embedding model
"""
import torchvision.datasets as dset
from tqdm import tqdm
import torch.multiprocessing
import sys
import torch
from losses.PILosses import PILossOBJ
from losses.CategoryLoss import LossCAT
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from utils.trainLogger_update import TrainingLogger
from models.VGG_PAN_SingleEmb import SingleModel
from models.VGG_PIE_SingleEmb import VGG_avg_picnn, VGG_avg_piproxy, VGG_avg_pitc
from utils.DataUtility import OOWLTrainDataset, MNet40TrainDataset, FG3DTrainDataset, OWSCTrainDataset, calculate_stats
from ConfigLearn import ConfigOOWL, ConfigMNet40, ConfigFG3D, ConfigOWSC_SI, ConfigOWSC_GN, HyperParams
from utils.InferenceUtility_PI import evaluate_performance_single
from utils.InferenceUtility_GN import evaluate_SI_performance_single # need to think about it
from utils.InferenceUtility_SI import evaluate_SI_performance_single # need to think about it
torch.multiprocessing.set_sharing_strategy('file_system')
print(torch.__version__)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""
VGG_PIE_SingleEmb: Pose Invariant Embedding for single embedding model
VGG_PAN_SingleEmb: Pose Invariant Attention Network for single embedding model
PILosses: Pose invariant losses (object loss & category loss)
CategoryLoss: Large margin softmax loss (category)
DataUtility: Custom DataUtilities (dataloader) for different datasets and other
             utility functions
InferenceUtility_PI: Functions for inference and computation of 
                     pose-invariant recognition accuracy and retrieval mAP
InferenceUtility_SI: For state invariant recognition accuracy and retrieval mAP
InferenceUtility_GN: For generalized recognition accuracy and retrieval mAP
ConfigLearn: Training and Testing Configurations for different datasets

"""

