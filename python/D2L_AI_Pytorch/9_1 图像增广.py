import time
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from matplotlib import pyplot as plt
from PIL import Image

import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

d2l.set_figsize()
img = Image.open('/Users/nihaopeng/Pictures/图库/20160930194558_zMLwj.jpeg')
# plt.imshow(img)
# plt.waitforbuttonpress(0)

def show_images(imgs, num_rows, num_cols, scale=2):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i * num_cols + j])
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    return axes

def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    show_images(Y, num_rows, num_cols, scale)

# apply(img, torchvision.transforms.RandomHorizontalFlip())       # 水平翻转
# apply(img, torchvision.transforms.RandomVerticalFlip())         # 上下翻转
# 亮度（brightness）、对比度（contrast）、饱和度（saturation）和色调（hue）
color_aug = torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
# 每次随机裁剪出一块面积为原面积10%∼100%的区域，且该区域的宽和高之比随机取自0.5∼2，然后再将该区域的宽和高分别缩放到200像素
shape_aug = torchvision.transforms.RandomResizedCrop(200, scale=(0.1, 1), ratio=(0.5, 2)) 
# 组合
augs = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(), color_aug, shape_aug])
apply(img, augs)

#############################################################################################################
all_imges = torchvision.datasets.CIFAR10(train=True, root="/Users/nihaopeng/Datasets/CIFAR", download=True)
show_images([all_imges[i][0] for i in range(32)], 4, 8, scale=0.8)
plt.imshow(img)
plt.waitforbuttonpress(0)

flip_aug = torchvision.transforms.Compose([
     torchvision.transforms.RandomHorizontalFlip(),
     torchvision.transforms.ToTensor()])

no_aug = torchvision.transforms.Compose([
     torchvision.transforms.ToTensor()])

num_workers = 0 if sys.platform.startswith('mac') else 4
def load_cifar10(is_train, augs, batch_size, root="/Users/nihaopeng/Datasets/CIFAR"):
    dataset = torchvision.datasets.CIFAR10(root=root, train=is_train, transform=augs, download=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=num_workers)

def train_with_data_aug(train_augs, test_augs, lr=0.001):
    batch_size, net = 256, d2l.resnet18(10)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = torch.nn.CrossEntropyLoss()
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    d2l.train(train_iter, test_iter, net, loss, optimizer, device, num_epochs=10)

# train_with_data_aug(flip_aug, no_aug)