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



class Resnet50(nn.Module):
	def __init__(self):
		super(Resnet50, self).__init__()

		self.cnn_layers = nn.Sequential(
			models.wide_resnet50_2(weights = models.Wide_ResNet50_2_Weights.IMAGENET1K_V2, progress = True)
		)

		self.fc_layers = nn.Sequential(
			nn.Linear(1000, 50)
		)

	def forward(self, x):

		x = self.cnn_layers(x)
		x = x.flatten(1)
		x = self.fc_layers(x)
		
		return x

class Densenet161(nn.Module):
	def __init__(self):
		super(Densenet161, self).__init__()

		self.CnnLayers = nn.Sequential(
			models.densenet161(pretrained = True)
		)

		self.FcLayer1 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Linear(1000, 512),
			nn.ReLU()
		)

		self.FcLayer2 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Linear(512, 256),
			nn.ReLU()
		)

		self.FcLayer3 = nn.Sequential(
			nn.Linear(256, 50)
		)

	def forward(self, x):

		x = self.CnnLayers(x)
		# print("x.shape = ", x.shape)
		x = x.flatten(1)
		x = self.FcLayer1(x)
		x = self.FcLayer2(x)
		x = self.FcLayer3(x)
		return x


class Dataset(data.Dataset):
	def __init__(self, img_list, label_list, transform_set, mode):
		self.img_list = img_list
		self.label_list = label_list
		self.transform_set = transform_set
		self.mode = mode

	def __getitem__(self, idx):
		img_path = self.img_list[idx][0]
		original_img = Image.open(img_path)
		transform = self.transform_set[int(self.img_list[idx][1])]
		transformed_img = transform(original_img)
		label = 0

		if(self.mode == "test"):
			label = 0
		else:
			label = self.label_list[idx]
		return transformed_img, label

	def __len__(self):
		return len(self.img_list)