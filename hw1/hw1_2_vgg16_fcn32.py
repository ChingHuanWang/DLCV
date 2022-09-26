import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder
from torchvision.transforms import transforms
import torchvision.models as models
from PIL import Image
import glob
import argparse
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import random


myseed = 6666  
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)


