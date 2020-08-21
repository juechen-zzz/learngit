import torch
from torch import nn

def comp_conv2d(conv2d, X):
    X = X.view((1, 1) + X.shape)
    # print(X.shape)                  # torch.Size([1, 1, 8, 8])
    Y = conv2d(X)
    return Y.view(Y.shape[2:])      # 排除不关心的前两维：批量和通道

# 注意这里是两侧分别填充1行或列，所以在两侧一共填充2行或列
conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)

X = torch.rand(8, 8)
print(comp_conv2d(conv2d, X).shape)

# 使用高为5、宽为3的卷积核。在高和宽两侧的填充数分别为2和1
conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5, 3), padding=(2, 1), stride=(4, 3))
print(comp_conv2d(conv2d, X).shape)

conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
print(comp_conv2d(conv2d, X).shape)

Y = torch.rand(3, 3)
conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, padding=1, stride=(3, 2))
print(comp_conv2d(conv2d, Y).shape)
