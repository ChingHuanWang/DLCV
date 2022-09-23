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


class Dataset(data.Dataset):
	def init(self, img_list, transform, random_noise, mode):
		self.img_list = img_list
		self.transform = transform
		self.random_noise = random_noise
		self.mode = mode

	def __getitem__(self, idx):
		img_path = self.img_list[idx]
		original_img = Image.open(img_path)
		transformed_img = self.transform(original_img)
		label = 0

		if(self.mode == "test"):
			label = 0
		else:
			label = int(img_path.split("/"[-1].split("_")[0]))
			if(self.mode == "train" and random.random() > 0.5):
				transformed_img = self.random_noise(transformed_img)

		return transformed_img, label

	def __len__():
		return len(self.img_list)
