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
import module


# **********************************************************
# **********************************************************
# 						testing part 
# **********************************************************
# **********************************************************
test_transform = transforms.Compose([
	transforms.Resize((32, 32)),
	transforms.ToTensor(),
	transforms.Normalize((0.5077, 0.4813, 0.4312), (0.2000, 0.1986, 0.2034))
])

test_list = glob.glob(os.path.join(test_path, "*.png"))
test_imgs = module.Dataset(test_list, transform = test_transform, random_noise = None, mode = "test")
test_loader = DataLoader(dataset = test_imgs, batch_size = 18, shuffle = False, num_workers = 0, pin_memeory = True)

bestModel = DenseNet161().to(device)
bestModel.load_state_dict(torch.load(model_path)["model_state_dict"])
bestModel.eval()
prediction = []
csv_file_path = ""

with torch.no_grad():
	for data, _ in test_loader:
		test_pred = bestModel(data.to(device))
		test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
		prediction += test_label.squeeze().tolist()

def pad4(i):
	return "0"*(4-len(str(i)))+str(i)

df = pd.DataFrame()
df["image_id"] = [pad4(i)+".png" for i in range(0,len(test_set))]
df["label"] = prediction
df.to_csv(csv_file_path, index = False)