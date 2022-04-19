# import packages
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


# define dataset
class SegDataset(data.Dataset):
  def __init__(self, inputs_path: list, labels_path=None, mode='train', transforms=None):
    self.inputs_path = inputs_path
    self.labels_path = labels_path
    self.transforms = transforms
    self.mode = mode
  
  def __len__(self):
    return len(self.inputs_path)
  
  def __getitem__(self, index: int):

    if self.mode == 'test':
        input_path = self.inputs_path[index]
        label = 0

        input = Image.open(input_path)

        if self.transforms is not None:
            transform = transforms.ToTensor()
            if random.random() > 0.5:
                input = tvF.hflip(input)
            
            input = self.transforms(input)

        else:
            transform = transforms.ToTensor()
            input = transform(input)
        
        return input, label

    else:
        input_path = self.inputs_path[index]
        label_path = self.labels_path[index]

        input = Image.open(input_path)
        label = Image.open(label_path)

        if self.transforms is not None:
            transform = transforms.ToTensor()
            if random.random() > 0.5:
                input = tvF.hflip(input)
                label = tvF.hflip(label)
            
            input, label = self.transforms(input), transform(label)

        else:
            transform = transforms.ToTensor()
            input, label = transform(input), transform(label)
        
        return input, label

# seed
def fix_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# get files path
def get_dataset(test_repo='./hw1_data/p2_data/validation', batch_size=4, mode='train'):
    path = "./hw1_data/p2_data/"
    train_files_path = os.path.join(path, 'train')
    test_files_path = test_repo

    split = [4, 9]
    train_files = glob.glob(os.path.join(train_files_path, '*'))
    valid_files = [train_file for train_file in train_files if (int(train_file.split('/')[-1].split('_')[0]) % 10) in split]
    train_files = [train_file for train_file in train_files if (int(train_file.split('/')[-1].split('_')[0]) % 10) not in split]
    test_files = glob.glob(os.path.join(test_files_path, '*'))

    train_inputs_files = [train_file for train_file in train_files if train_file.split('.')[-1] == "jpg"]
    train_labels_files = [train_file for train_file in train_files if train_file.split('.')[-1] == "png"]

    valid_inputs_files = [valid_file for valid_file in valid_files if valid_file.split('.')[-1] == "jpg"]
    valid_labels_files = [valid_file for valid_file in valid_files if valid_file.split('.')[-1] == "png"]

    if mode == 'train':
        test_inputs_files = [test_file for test_file in test_files if test_file.split('.')[-1] == "jpg"]
        test_labels_files = [test_file for test_file in test_files if test_file.split('.')[-1] == "png"]    
    else:
        test_inputs_files = [test_file for test_file in test_files if test_file.split('.')[-1] == "jpg"]

    train_inputs_files.sort()
    train_labels_files.sort()
    valid_inputs_files.sort()
    valid_labels_files.sort()
    if mode == 'train':
        test_inputs_files.sort()
        test_labels_files.sort()
    else:
        test_inputs_files.sort()

    with open('./check2.txt', 'a') as f:
        f.write(f"{len(train_inputs_files)}\n")
        f.write(f"{train_inputs_files[:10]}\n")
        f.write(f"{train_labels_files[:10]}\n")
        f.write(f"{len(valid_inputs_files)}\n")
        f.write(f"{len(test_inputs_files)}\n")

    # transforms
    train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # dataloader
    train_set = SegDataset(train_inputs_files, train_labels_files, mode='train', transforms=train_transforms)
    valid_set = SegDataset(valid_inputs_files, valid_labels_files, mode='valid', transforms=test_transforms)
    if mode == 'train':
        test_set = SegDataset(test_inputs_files, test_labels_files, mode='valid', transforms=test_transforms)
    else:
        test_set = SegDataset(test_inputs_files, mode='test', transforms=test_transforms)

    n_workers = 0
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader, test_inputs_files

def get_test_dataset(test_repo='./hw1_data/p2_data/validation', batch_size=4, mode='train'):
    # path = "./hw1_data/p2_data/"
    # train_files_path = os.path.join(path, 'train')
    test_files_path = test_repo

    # split = [4, 9]
    # train_files = glob.glob(os.path.join(train_files_path, '*'))
    # valid_files = [train_file for train_file in train_files if (int(train_file.split('/')[-1].split('_')[0]) % 10) in split]
    # train_files = [train_file for train_file in train_files if (int(train_file.split('/')[-1].split('_')[0]) % 10) not in split]
    test_files = glob.glob(os.path.join(test_files_path, '*'))

    # train_inputs_files = [train_file for train_file in train_files if train_file.split('.')[-1] == "jpg"]
    # train_labels_files = [train_file for train_file in train_files if train_file.split('.')[-1] == "png"]

    # valid_inputs_files = [valid_file for valid_file in valid_files if valid_file.split('.')[-1] == "jpg"]
    # valid_labels_files = [valid_file for valid_file in valid_files if valid_file.split('.')[-1] == "png"]

    # if mode == 'train':
    #     test_inputs_files = [test_file for test_file in test_files if test_file.split('.')[-1] == "jpg"]
    #     test_labels_files = [test_file for test_file in test_files if test_file.split('.')[-1] == "png"]    
    # else:
    test_inputs_files = [test_file for test_file in test_files if test_file.split('.')[-1] == "jpg"]

    # train_inputs_files.sort()
    # train_labels_files.sort()
    # valid_inputs_files.sort()
    # valid_labels_files.sort()
    # if mode == 'train':
    #     test_inputs_files.sort()
    #     test_labels_files.sort()
    # else:
    test_inputs_files.sort()

    # with open('./check2.txt', 'a') as f:
    #     f.write(f"{len(train_inputs_files)}\n")
    #     f.write(f"{train_inputs_files[:10]}\n")
    #     f.write(f"{train_labels_files[:10]}\n")
    #     f.write(f"{len(valid_inputs_files)}\n")
    #     f.write(f"{len(test_inputs_files)}\n")

    # transforms
    # train_transforms = transforms.Compose([
    # transforms.ToTensor(),
    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # ])

    test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # dataloader
    # train_set = SegDataset(train_inputs_files, train_labels_files, mode='train', transforms=train_transforms)
    # valid_set = SegDataset(valid_inputs_files, valid_labels_files, mode='valid', transforms=test_transforms)
    # if mode == 'train':
    #     test_set = SegDataset(test_inputs_files, test_labels_files, mode='valid', transforms=test_transforms)
    # else:
    test_set = SegDataset(test_inputs_files, mode='test', transforms=test_transforms)

    n_workers = 0
    # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=True)
    # valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return test_loader, test_inputs_files

# define model
class vgg16_fcn32(nn.Module):
  def __init__(self, num_classes):
    super().__init__()

    self.vgg = models.vgg16(pretrained=True).features

    self.fcn = nn.Sequential(
        nn.ConvTranspose2d(512, 256, 4, 2, padding=1),
        nn.ReLU(inplace=True),
        nn.Dropout(),

        nn.ConvTranspose2d(256, 128, 4, 2, padding=1),
        nn.ReLU(inplace=True),
        nn.Dropout(),

        nn.ConvTranspose2d(128, 64, 4, 2, padding=1),
        nn.ReLU(inplace=True),
        nn.Dropout(),

        nn.ConvTranspose2d(64, 32, 4, 2, padding=1),
        nn.ReLU(inplace=True),
        nn.Dropout(),

        nn.ConvTranspose2d(32, num_classes, 4, 2, padding=1)
    )
  def forward(self, x):
    x = self.vgg(x)
    x = self.fcn(x)
    return x

class Net(nn.Module):
  def __init__(self, num_classes):
    super().__init__()

    resnet_50 = models.resnet50(pretrained=True)

    self.resnet = nn.Sequential(
        resnet_50.conv1,
        resnet_50.bn1,
        resnet_50.relu,
        resnet_50.maxpool,
        resnet_50.layer1,
        resnet_50.layer2,
        resnet_50.layer3,
        resnet_50.layer4,
    )

    def decode(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
      dec = nn.Sequential(
          nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
          nn.BatchNorm2d(out_channels),
          nn.ReLU(inplace=True)
      )
      return dec

    self.decode = nn.Sequential(
        decode(2048, 512, 4, 2, 1),
        decode(512, 512),
        decode(512, 512),
        decode(512, 128, 4, 2, 1),
        decode(128, 128),
        decode(128, 128),
        decode(128, 32, 4, 2, 1),
        decode(32, 32),
        decode(32, 16, 4, 2, 1),
        decode(16, 16),
        decode(16, 8, 4, 2, 1),
        decode(8, num_classes),
    )

  def forward(self, x):
    x = self.resnet(x)
    x = self.decode(x)
    return x

# mean_iou
def read_masks(data, image_size=512):
  # masks = np.empty((data.shape[0], image_size, image_size))
  masks = torch.rand((data.shape[0], image_size, image_size)) * 6
  masks = torch.floor(masks).numpy()
  # masks = (masks - np.min(masks)) / np.max(masks) * 6
  for i, d in enumerate(data):
    # mask = d.cpu().detach().numpy() * 255
    mask = d.cpu().detach().numpy()
    # mask = (mask >= 128).astype(int)
    mask = 4 * mask[0, :, :] + 2 * mask[1, :, :] + mask[2, :, :]
    masks[i, mask == 3] = 0  # (Cyan: 011) Urban land 
    masks[i, mask == 6] = 1  # (Yellow: 110) Agriculture land 
    masks[i, mask == 5] = 2  # (Purple: 101) Rangeland 
    masks[i, mask == 2] = 3  # (Green: 010) Forest land 
    masks[i, mask == 1] = 4  # (Blue: 001) Water 
    masks[i, mask == 7] = 5  # (White: 111) Barren land 
    masks[i, mask == 0] = 6  # (Black: 000) Unknown 
  return masks

def mean_iou_score(pred, labels):
    # print(pred, labels)
    # mean_iou = 0
    # count = 0
    tp_fps = []
    tp_fns = []
    tps = []
    for i in range(6):
      tp_fp = np.sum(pred == i)
      tp_fn = np.sum(labels == i)
      tp = np.sum((pred == i) * (labels == i))
      tp_fps.append(tp_fp)
      tp_fns.append(tp_fn)
      tps.append(tp)
    #   if (tp_fp + tp_fn - tp) == 0:
    #     count += 1
    #     # print(f'class {i}: 0')
    #   else:
    #     iou = tp / (tp_fp + tp_fn - tp)
    #     mean_iou += iou
    #     # print(f'class {i}: {iou}')
    # if (6 - count) != 0:
    #   mean_iou /= (6 - count)
    # print(f"mean_iou = {mean_iou}")
    return tp_fps, tp_fns, tps


# train
def train(device, model, train_loader, valid_loader, n_epochs=50, accum_steps=4):

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    max_valid_miou = 0

    if os.path.isfile('./vgg16fcn32.ckpt'):
        ckpt = torch.load('./vgg16fcn32.ckpt')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        # scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['last_epoch'] + 1
        max_valid_miou = ckpt['max_valid_miou']
    else:
        start_epoch = 0
        with open('./record2.txt', 'w') as f:
            f.write('')

    for epoch in range(start_epoch, n_epochs):

        model.train()
        train_loss = []
        train_fps = []
        train_fns = []
        train_tps = []
        optimizer.zero_grad()

        for index, batch in enumerate(train_loader):

            inputs, labels = batch
            
            pred = model(inputs.to(device))
            
            labels = read_masks(labels)
            labels = torch.from_numpy(labels).long().to(device)

            loss = loss_fn(pred, labels) / accum_steps
            loss.backward()

            tp_fps, tp_fns, tps = mean_iou_score(pred.argmax(dim=1).cpu().numpy(), labels.cpu().numpy())
            train_fps.append(tp_fps)
            train_fns.append(tp_fns)
            train_tps.append(tps)

            if ((index + 1) % accum_steps == 0) or ((index + 1) == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad()
            
            train_loss.append(loss.item())

        train_loss = sum(train_loss) / len(train_loss)
        train_fps = np.sum(np.array(train_fps), axis=0)
        train_fns = np.sum(np.array(train_fns), axis=0)
        train_tps = np.sum(np.array(train_tps), axis=0)
        train_mious = train_tps / (train_fps + train_fns - train_tps)
        train_miou = np.sum(train_mious) / len(train_mious)

        with open('./record2.txt', 'a') as f:
            f.write(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, mean iou = {train_miou:.5f}\n")
            
        
        model.eval()
        valid_fps = []
        valid_fns = []
        valid_tps = []
        valid_loss = []

        with torch.no_grad():
            for batch in valid_loader:
            
                inputs, labels = batch
                
                pred = model(inputs.to(device))
                
                labels = read_masks(labels)
                labels = torch.from_numpy(labels).long().to(device)
                loss = loss_fn(pred, labels)

                tp_fps, tp_fns, tps = mean_iou_score(pred.argmax(dim=1).cpu().numpy(), labels.cpu().numpy())
                valid_fps.append(tp_fps)
                valid_fns.append(tp_fns)
                valid_tps.append(tps)
                
                valid_loss.append(loss.item())
        
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_fps = np.sum(np.array(valid_fps), axis=0)
        valid_fns = np.sum(np.array(valid_fns), axis=0)
        valid_tps = np.sum(np.array(valid_tps), axis=0)
        valid_mious = valid_tps / (valid_fps + valid_fns - valid_tps)
        valid_miou = np.sum(valid_mious) / len(valid_mious)

        with open('./record2.txt', 'a') as f:
            f.write(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, mean iou = {valid_miou:.5f}\n")

        if max_valid_miou < valid_miou:
            torch.save({'last_epoch': epoch + 1,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'max_valid_miou': max_valid_miou,
                        # 'scheduler': scheduler.state_dict(),
                        }, f'./vgg16fcn32.ckpt')
            max_valid_miou = valid_miou
            with open('./record2.txt', 'a') as f:
                f.write('Saving model\n')
        
        with open('./record2.txt', 'a') as f:
            f.write(f"max_valid_miou = {max_valid_miou:.5f}\n")

  # scheduler.step()

# test
def test(device, model, test_loader, output_repo, test_inputs_files):
    # with open('./record2.txt', 'a') as f:
    #     f.write("testing\n")
    test_mious = []

    ckpt = torch.load(f'./model.ckpt', map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.eval()
    # test_fps = []
    # test_fns = []
    # test_tps = []
    preds = []

    for batch in test_loader:

        inputs, _ = batch
        
        with torch.no_grad():
            pred = model(inputs.to(device))
        
        # labels = read_masks(labels)
        # labels = torch.from_numpy(labels).long().to(device)
        preds.append(pred.argmax(dim=1).cpu().numpy())

        # tp_fps, tp_fns, tps = mean_iou_score(pred.argmax(dim=1).cpu().numpy(), labels.cpu().numpy())
        # test_fps.append(tp_fps)
        # test_fns.append(tp_fns)
        # test_tps.append(tps)

    # test_fps = np.sum(np.array(test_fps), axis=0)
    # test_fns = np.sum(np.array(test_fns), axis=0)
    # test_tps = np.sum(np.array(test_tps), axis=0)
    # test_mious = test_tps / (test_fps + test_fns - test_tps)
    # test_miou = np.sum(test_mious) / len(test_mious)

    # with open('./record2.txt', 'a') as f:
    #     f.write(f"test miou = {test_miou:.5f}\n")

    for j, pred in enumerate(preds):
        masks = np.zeros((pred.shape[0], 512, 512, 3))
        for index, i in enumerate(pred):
            masks[index, i == 0, 2] = 1
            masks[index, i == 2, 2] = 1
            masks[index, i == 4, 2] = 1
            masks[index, i == 5, 2] = 1

            masks[index, i == 0, 1] = 1
            masks[index, i == 1, 1] = 1
            masks[index, i == 3, 1] = 1
            masks[index, i == 5, 1] = 1

            masks[index, i == 1, 0] = 1
            masks[index, i == 2, 0] = 1
            masks[index, i == 5, 0] = 1

            fn = test_inputs_files[j * 4 + index].split('/')[-1].split('.')[0] + '.png'
            output_path = os.path.join(output_repo, fn)
            plt.imsave(output_path, masks[index])

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True)
    ap.add_argument('--test_repo', required=True)
    ap.add_argument('--output_repo', required=True)
    args = ap.parse_args()
    mode = args.mode
    test_repo = args.test_repo
    output_repo = args.output_repo

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 0
    fix_random_seeds(seed)

    batch_size = 4
    num_classes = 7
    model = Net(num_classes=num_classes).to(device)

    if mode == 'train':
        train_loader, valid_loader, test_loader, test_inputs_files = get_dataset(test_repo, batch_size, mode)
        train(device, model, train_loader, valid_loader, n_epochs=75, accum_steps=4)
    
    if mode == "test":
        test_loader, test_inputs_files = get_test_dataset(test_repo, batch_size, mode)
        test(device, model, test_loader, output_repo, test_inputs_files)