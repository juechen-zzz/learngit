import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

# mnist_train和mnist_test都是torch.utils.data.Dataset的子类，所以我们可以用len()来获取该数据集的大小，还可以用下标来获取具体的一个样本。
# 训练集中和测试集中的每个类别的图像数分别为6,000和1,000。因为有10个类别，所以训练集和测试集的样本数分别为60,000和10,000。
mnist_train = torchvision.datasets.FashionMNIST(root='/Users/nihaopeng/个人/Git/learngit/python/D2L_AI_Pytorch', train=True, download=False, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='/Users/nihaopeng/个人/Git/learngit/python/D2L_AI_Pytorch', train=False, download=False, transform=transforms.ToTensor())

print(type(mnist_train))
print(len(mnist_train), len(mnist_test))

feature, label = mnist_train[0]
print(feature.shape, label)  # Channel x Height x Width

batch_size = 256
num_workers = 0
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# 查看遍历时间
start = time.time()
for X, y in train_iter:
    continue
print('%.2f sec' % (time.time() - start))