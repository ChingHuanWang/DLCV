import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.utils import data
from torchvision.transforms import transforms
import torchvision.models as models
import torchvision.transforms.functional as tvF
from PIL import Image
import glob
import random
import argparse
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# sat -> jpg, mask -> png


train_sat_img_path_list = glob.glob(os.path.join("./hw1_data/p2_data/train", "*.jpg"))
train_mask_img_path_list = glob.glob(os.path.join("./hw1_data/p2_data/train", "*.png"))
test_sat_img_path_list = glob.glob(os.path.join("./hw1_data/p2_data/val", ".jpg"))


transform = transforms.Compose([
	transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# print(training_sat_img_path_list)

class Dataset(data.Dataset):
	def __init__(self, sat_img_path_list, mask_img_path_list=None, mode, transform):
		self.sat_img_path_list = sat_img_path_list
		self.mask_img_path_list = mask_img_path_list
		self.transform = transform
		self.mode = mode

	def __len__(self):
		return len(self.sat_img_path_list)

	def __getitem__(self, idx):

		if(self.mode == "test"):
			sat = Image.open(self.sat_img_path_list[idx])
			mask = 0

			if(self.transform == None):

				transform = transforms.ToTensor()
				sat = transform(sat)

			else:

				sat = self.transform(sat)

			return sat, mask

		else:

			sat = Image.open(self.sat_img_path_list[idx])
			mask = Image.open(self.mask_img_path_list[idx])

			if(self.transform == None):
				transform = transforms.ToTensor()
				sat, mask = transform(sat), transform(mask)

			else:

				sat, mask = self.transform(sat), self.transform(mask)

			return sat, mask


train_sat_img_path_list, val_sat_img_path_list, train_mask_img_path_list, val_mask_img_path_list = train_test_split(train_sat_img_path_list, train_mask_img_path_list, train_size = 0.8, random_state = 2022)

train_data = Dataset(train_sat_img_path_list, train_mask_img_path_list, "train", transform)
val_data = Dataset(val_sat_img_path_list, val_mask_img_path_list, "val", transform)
test_data = Dataset(test_sat_img_path_list, None, "test", transform)

class VGG16_FCN32(nn.Module):
	def __init__(self, num_class = 7):
		super().__init__()

		self.vgg16 = models.vgg16(pretrained = True).features

		self.fcn32 = nn.Sequential(
			# conv1
			nn.Conv2d(3, 64, 3, padding=100)
	        nn.ReLU(inplace=True)
	        nn.Conv2d(64, 64, 3, padding=1)
	        nn.ReLU(inplace=True)
	        nn.MaxPool2d(2, stride=2, ceil_mode=True)

	        nn.Conv2d(64, 128, 3, padding=1)
	        nn.ReLU(inplace=True)
	        nn.Conv2d(128, 128, 3, padding=1)
	        nn.ReLU(inplace=True)
	        nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

	        nn.Conv2d(128, 256, 3, padding=1)
        	nn.ReLU(inplace=True)
        	nn.Conv2d(256, 256, 3, padding=1)
	        nn.ReLU(inplace=True)
	        nn.Conv2d(256, 256, 3, padding=1)
	        nn.ReLU(inplace=True)
	        nn.MaxPool2d(2, stride=2, ceil_mode=True)

	        nn.Conv2d(256, 512, 3, padding=1)
        	nn.ReLU(inplace=True)
        	nn.Conv2d(512, 512, 3, padding=1)
        	nn.ReLU(inplace=True)
        	nn.Conv2d(512, 512, 3, padding=1)
        	nn.ReLU(inplace=True)
        	nn.MaxPool2d(2, stride=2, ceil_mode=True)

        	nn.Conv2d(512, 512, 3, padding=1)
	        nn.ReLU(inplace=True)
	        nn.Conv2d(512, 512, 3, padding=1)
	        nn.ReLU(inplace=True)
	        nn.Conv2d(512, 512, 3, padding=1)
	        nn.ReLU(inplace=True)
	        nn.MaxPool2d(2, stride=2, ceil_mode=True)

	        nn.Conv2d(512, 4096, 7)
	        nn.ReLU(inplace=True)
	        nn.Dropout2d()

	        nn.Conv2d(4096, 4096, 1)
	        nn.ReLU(inplace=True)
	        nn.Dropout2d()

	        nn.Conv2d(4096, n_class, 1)
	        nn.ConvTranspose2d(n_class, n_class, 64, stride=32, bias=False)
		)

	def forward(self, x):

		out = self.vgg16(x)
		out = self.fcn32(out)
		return out