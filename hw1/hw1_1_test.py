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
import pandas as pd

from hw1_1_model import Resnet50
from hw1_1_model import Densenet161
from hw1_1_model import Dataset


# **********************************************************
# **********************************************************
# 						testing part 
# **********************************************************
# **********************************************************
csv_file_path = "densenet161_prediction.csv"
model_path = "densenet161_model.pt"
test_path = "./p1_data/val_50"

def sorted_img_and_label_list(img_list, label_list):
    sorted_idxs = np.argsort(label_list)
    label_list = np.array(label_list)[sorted_idxs]
    img_list = np.array(img_list)[sorted_idxs]

    return img_list, label_list

test_transform = transforms.Compose([
	transforms.Resize((224, 224)),
	transforms.ToTensor(),
	transforms.Normalize((0.5077, 0.4813, 0.4312), (0.2000, 0.1986, 0.2034))
])

transform_set = [test_transform]

test_img_list = glob.glob(os.path.join(test_path, "*.png"))
test_label_list = np.array([int(img.split("/")[-1].split("_")[0]) for img in test_img_list])
test_img_list = np.array([[img, 0] for img in test_img_list])

test_img_list, test_label_list = sorted_img_and_label_list(test_img_list, test_label_list)
test_imgs = Dataset(test_img_list, test_label_list, transform_set = transform_set, mode = "test")
test_loader = DataLoader(dataset = test_imgs, batch_size = 18, shuffle = False)


device = "cuda" if torch.cuda.is_available() else "cpu"
bestModel = Densenet161().to(device)
bestModel.load_state_dict(torch.load(model_path)["model_state_dict"])
bestModel.eval()
prediction = []





# with open("test.txt", "a") as f:
# 	for img in test_img_list:
# 		f.write(f"path = {img[0]}\n")

with torch.no_grad():
	for data, _ in tqdm(test_loader):
		test_pred = bestModel(data.to(device))
		test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
		prediction += test_label.squeeze().tolist()

def pad4(i):
	return "0"*(4-len(str(i)))+str(i)

df = pd.DataFrame()
df["image_id"] = [img[0].split("/")[-1]+".png" for img in test_img_list]
df["label"] = prediction
df.to_csv(csv_file_path, index = False)



