import torch
from torch import nn
import sys
sys.path.append("..") 
import d2lzh_pytorch as d2l

def corr2d(X, K): 
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y

# 多输入通道
def corr2d_multi_in(X, K):
    res = corr2d(X[0, :, :], K[0, :, :])
    for i in range(1, X.shape[0]):
        res += corr2d(X[i, :, :], K[i, :, :])
    return res

# 对于RGB三通道图像，在指定kernel_size的前提下，真正的卷积核大小是kernel_size*kernel_size*3
# X.shape: 2 * 3 * 3   K.shape: 2 * 2 * 2
X = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
K = torch.tensor([[[0, 1], [2, 3]], [[1, 2], [3, 4]]])

print(corr2d_multi_in(X, K))    # torch.Size([2, 2])

# 多输出通道
def corr2d_multi_out(X, K):
    return torch.stack([corr2d_multi_in(X, k) for k in K])

K = torch.stack([K, K + 1, K + 2])
# print(K.shape)                # torch.Size([3, 2, 2, 2])
print(corr2d_multi_out(X, K))   # torch.Size([3, 2, 2])
