# 池化的目的是为了缓解卷积层对位置的过度敏感
# 在处理多通道输入数据时，池化层对每个输入通道分别池化，而不是像卷积层那样将各通道的输入按通道相加。
# 这意味着池化层的输出通道数与输入通道数相等。下面将数组X和X+1在通道维上连结来构造通道数为2的输入。

import torch
from torch import nn

def pool2d(X, pool_size, mode='max'):
    X = X.float()
    p_h, p_w = pool_size
    Y = torch.zeros(X.shape[0]-p_h+1, X.shape[1]-p_w+1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i+p_h, j: j+p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i+p_h, j: j+p_w].mean()
    return Y

X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
print(pool2d(X, (2, 2)))
print(pool2d(X, (2, 2), 'avg'))

X = torch.arange(16, dtype=torch.float).view((1, 1, 4, 4))
print(X)
pool2d = nn.MaxPool2d(3)    # 默认情况下，MaxPool2d实例里步幅和池化窗口形状相同
print(pool2d(X))
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print(pool2d(X))
pool2d = nn.MaxPool2d((2, 4), padding=(1, 2), stride=(2, 3))
print(pool2d(X))            # [(w - k + p + s) / s ] * [(h - k + p + s) / s]    p = 2 * padding