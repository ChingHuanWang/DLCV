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
# import module


# set a random seed for reproducibility
myseed = 6666  
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)



class Dataset(data.Dataset):
	def __init__(self, img_list, label_list, transform, random_noise, mode):
		self.img_list = img_list
		self.label_list = label_list
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
			label = self.label_list[idx]
			if(self.mode == "train" and random.random() > 0.5):
				transformed_img = self.random_noise(transformed_img)

		return transformed_img, label

	def __len__(self):
		return len(self.img_list)


train_path = "./p1_data/train_50"
model_path = "model.pt"
record_path = "record.txt"


# let data distribution is even 
train_img_list = glob.glob(os.path.join(train_path, "*.png"))
train_label_list = [int(img.split("/")[-1].split("_")[0]) for img in train_img_list]
train_img_list, valid_img_list, train_label_list, valid_label_list = train_test_split(train_img_list, train_label_list, train_size = 0.8, random_state = 2022, stratify = train_label_list)



# train_transform = transforms.Compose([
# 	transforms.Resize((32, 32)),
# 	transforms.ToTensor(),
# 	transforms.Normalize((0.5077, 0.4813, 0.4312), (0.2000, 0.1986, 0.2034)),
# 	transforms.RandomHorizontalFlip(p = 0.25),
# 	transforms.RandomInvert(p = 0.25),
# 	transforms.RandomGrayscale(p = 0.25),
# 	transforms.RandomAdjustSharpness(10, p = 0.25),
# 	transforms.ColorJitter(brightness = (0.5, 1.5), contrast = (0.5, 1.5), saturation = (0.5, 1.5), hue = (-0.1, 0.1))
# ])

train_transform = transforms.Compose([
	transforms.Resize((224, 224)),
	transforms.ToTensor(),
	transforms.Normalize((0.5077, 0.4813, 0.4312), (0.2000, 0.1986, 0.2034)),
])


valid_transform = transforms.Compose([
	transforms.Resize((224, 224)),
	transforms.ToTensor(),
	transforms.Normalize((0.5077, 0.4813, 0.4312), (0.2000, 0.1986, 0.2034))
])

random_noise = transforms.GaussianBlur(kernel_size = (3, 3), sigma = (5, 5))


train_set = Dataset(img_list = train_img_list, label_list = train_label_list, transform = train_transform, random_noise = random_noise, mode = "train")
valid_set = Dataset(img_list = valid_img_list, label_list = valid_label_list, transform = valid_transform, random_noise = None, mode = "valid")


train_loader = DataLoader(dataset = train_set, batch_size = 18, shuffle = True, num_workers = 0, pin_memory = True)
valid_loader = DataLoader(dataset = valid_set, batch_size = 18, shuffle = True, num_workers = 0, pin_memory = True)



class DenseNet161(nn.Module):
	def __init__(self):
		super(DenseNet161, self).__init__()

		self.CnnLayers = nn.Sequential(
			models.densenet161(pretrained = False)
		)

		self.FcLayers = nn.Sequential(
			nn.Linear(1000, 256),
			nn.ReLU(),
			nn.Linear(256, 64),
			nn.ReLU(),
			nn.Linear(64, 50)
		)

	def forward(self, x):

		x = self.CnnLayers(x)
		x = x.flatten(1)
		x = self.FcLayers(x)
		return x


device = "cuda" if torch.cuda.is_available() else "cpu"
model = DenseNet161().to(device)
model.device = device
prev_epoch = 0
max_valid_acc = 0

if(os.stat(model_path).st_size != 0):
	checkPoint = None
	if(torch.cuda.is_available()):
		checkPoint = torch.load(model_path)

	else:
		checkPoint = torch.load(model_path, map_location = "cpu")

	model.load_state_dict(checkPoint["model_state_dict"])
	prev_epoch = checkPoint["epoch"]
	max_valid_acc = checkPoint["max_valid_acc"]

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0003, weight_decay = 1e-5)
n_epoch = 300
stale = 0
patience = 100

# **********************************************************
# **********************************************************
# 						training part 
# **********************************************************
# **********************************************************

for epoch in range(prev_epoch, n_epoch):

	model.train()
	train_loss = []
	train_accs = []

	for batch in tqdm(train_loader):
		
		imgs, labels = batch

		logits = model(imgs.to(device))

		loss = criterion(logits, labels.to(device))

		optimizer.zero_grad()

		loss.backward()

		grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm = 10)

		optimizer.step()

		acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

		train_loss.append(loss.item())
		train_accs.append(acc)

	train_loss = sum(train_loss) / len(train_loss)
	train_acc = sum(train_accs) / len(train_accs)

	print(f"[ Train | {epoch + 1:03d}/{n_epoch:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

	model.eval()

	valid_loss = []
	valid_accs = []

	for batch in tqdm(valid_loader):

		imgs, labels = batch

		with torch.no_grad():
			logits = model(imgs.to(device))

		loss = criterion(logits, labels.to(device))

		acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

		valid_loss.append(loss.item())
		valid_accs.append(acc)

		valid_loss.append(loss.item())
		valid_accs.append(acc)

	valid_loss = sum(valid_loss) / len(valid_loss)
	valid_acc = sum(valid_accs) / len(valid_accs)

	print(f"[ Valid | {epoch + 1:03d}/{n_epoch:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

	if valid_acc > max_valid_acc:
		with open(record_path,"a") as f:
			f.write(f"Best model found at epoch {epoch}, saving model\n")
			f.write(f"[ Valid | {epoch + 1:03d}/{n_epoch:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best\n")
	else:
		with open(record_path,"a") as f:
			f.write(f"[ Valid | {epoch + 1:03d}/{n_epoch:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}\n")

	if valid_acc > max_valid_acc:
		print(f"Best model found at epoch {epoch}, saving model")
		torch.save({"model_state_dict" : model.state_dict(),
			"epoch" : epoch,
			"max_valid_acc" : valid_acc
		}, model_path)

		max_valid_acc = valid_acc
		stale = 0
	else:
		stale += 1
		if stale > patience:
			print(f"No improvment {patience} consecutive epochs, early stopping")
			break

