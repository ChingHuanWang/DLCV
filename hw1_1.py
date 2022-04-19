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


class Dataset(data.Dataset):
  def __init__(self, inputs_path: list, mode, transforms=None):
    self.inputs_path = inputs_path
    self.transforms = transforms
    self.mode = mode
  
  def __len__(self):
    return len(self.inputs_path)
  
  def __getitem__(self, index: int):
    input_path = self.inputs_path[index]
    input = Image.open(input_path)
    if self.mode == 'test':
      label = 0
    else:
      label = int(input_path.split('/')[-1].split('_')[0])
    

    if self.transforms is not None:
      input = self.transforms(input)

    else:
      transform = transforms.ToTensor()
      input = transform(input)
    
    return input, label


def fix_random_seeds(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def get_dataset(batch_size, test_repo):
    path = "./hw1_data/p1_data"
    test_path = os.path.join(test_repo, '*.png')
    train_path = os.path.join(path, 'train_50/*png')

    # split training files and get testing files path
    split = [4, 9]
    train_files = glob.glob(train_path)
    valid_files = [train_file for train_file in train_files if int(train_file.split('_')[-1].split('.')[0]) % 10 in split]
    train_files = [train_file for train_file in train_files if int(train_file.split('_')[-1].split('.')[0]) % 10 not in split]
    test_files = glob.glob(test_path)

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop((96, 96)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_set = Dataset(train_files, 'train', transforms=train_transforms)
    valid_set = Dataset(valid_files, 'valid', transforms=test_transforms)
    test_set = Dataset(test_files, 'test', transforms=test_transforms)

    with open("./check.txt", 'a') as f:
        f.write(f"{len(train_set)}\n")
        f.write(f"{len(valid_set)}\n")
        f.write(f"{len(test_set)}\n")

    batch_size = 32
    n_workers = 0
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader, test_files

def get_test_dataset(batch_size, test_repo):
    # path = "./hw1_data/p1_data"
    test_path = os.path.join(test_repo, '*.png')
    # train_path = os.path.join(path, 'train_50/*png')

    # split training files and get testing files path
    # split = [4, 9]
    # train_files = glob.glob(train_path)
    # valid_files = [train_file for train_file in train_files if int(train_file.split('_')[-1].split('.')[0]) % 10 in split]
    # train_files = [train_file for train_file in train_files if int(train_file.split('_')[-1].split('.')[0]) % 10 not in split]
    test_files = glob.glob(test_path)

    # train_transforms = transforms.Compose([
    #     transforms.RandomResizedCrop((96, 96)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # ])

    test_transforms = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # train_set = Dataset(train_files, 'train', transforms=train_transforms)
    # valid_set = Dataset(valid_files, 'valid', transforms=test_transforms)
    test_set = Dataset(test_files, 'test', transforms=test_transforms)

    # with open("./check.txt", 'a') as f:
    #     f.write(f"{len(train_set)}\n")
    #     f.write(f"{len(valid_set)}\n")
    #     f.write(f"{len(test_set)}\n")

    batch_size = 32
    n_workers = 0
    # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=True)
    # valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return test_loader, test_files


def train(device, model, train_loader, valid_loader, n_epochs=25, accum_steps=10):

    # train model
    model.device = device

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    max_valid_acc = 0

    for epoch in range(n_epochs): 

        model.train()
        train_loss = []
        train_accs = []
        optimizer.zero_grad()

        for i, batch in enumerate(train_loader):

            imgs, labels = batch
            
            logits = model(imgs.to(device))
            loss = criterion(logits, labels.to(device)) / accum_steps
            # loss = criterion(logits, labels.to(device))
            loss.backward()
            if ((i + 1) % accum_steps == 0) or ((i + 1) == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad()
            
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
            train_loss.append(loss.item())
            train_accs.append(acc)
        
        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)  
        
        model.eval()
        valid_loss = []
        valid_accs = []
        with torch.no_grad():
            for batch in valid_loader:

                    imgs, labels = batch

                    logits = model(imgs.to(device))
                    loss = criterion(logits, labels.to(device))
                    acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

                    valid_loss.append(loss.item())
                    valid_accs.append(acc)

        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)

        if max_valid_acc < valid_acc:
            torch.save(model.state_dict(), f'./model.pth')
            max_valid_acc = valid_acc
            with open('./record.txt', 'a') as f:
                f.write(f"saving model\n")
        
        with open('./record.txt', 'a') as f:
            f.write(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}\n")
            f.write(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}\n")
            f.write(f"max_valid_acc = {max_valid_acc:.5f}\n")
        
        scheduler.step()

def test(model, test_loader, test_files, csv_path='./predict.csv'):

    # with open('./record.txt', 'a') as f:
    #     f.write("testing\n")

    # test_accs = []
    predictions = []
    with torch.no_grad():
        for batch in test_loader:

            imgs, _ = batch

            best_model = torch.load(f'./model.pth', map_location='cpu')
            model.load_state_dict(best_model)
            model.eval()
                
            logits = model(imgs.to(device))

            # logits = torch.zeros(logitss[0].shape).to(device)
            # for index, logit in enumerate(logitss):
            #     logits += logit / len(resnet_models)
            
            # acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            # test_accs.append(acc)
            predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())

    # test_acc = sum(test_accs) / len(test_accs)
    # with open('./record.txt', 'a') as f:
    #     f.write(f"test_acc = {test_acc:.5f}\n")

    file_id = []
    for test_file in test_files:
        file_id.append(test_file.split('/')[-1])

    with open(csv_path, "w") as f:
        f.write("image_id,label\n")
        for i, pred in enumerate(predictions):
            f.write(f"{file_id[i]},{pred}\n")



if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True)
    ap.add_argument('--test_repo', required=True)
    ap.add_argument('--csv_path', required=True)
    args = ap.parse_args()
    mode = args.mode
    test_repo = args.test_repo
    csv_path = args.csv_path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fix_random_seeds()

    batch_size = 32

    model = models.resnet152(pretrained=True)
    fc_input = model.fc.in_features
    model.fc = nn.Linear(fc_input, 50)
    model = model.to(device)

    if mode == 'train':
        train_loader, valid_loader, test_loader, test_files = get_dataset(batch_size, test_repo)
        train(device, model, train_loader, valid_loader, n_epochs=25, accum_steps=10)
    
    if mode == "test":
        test_loader, test_files = get_test_dataset(batch_size, test_repo)
        test(model, test_loader, test_files, csv_path)
