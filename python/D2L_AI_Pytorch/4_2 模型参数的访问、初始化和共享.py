import torch
from torch import nn
from torch.nn import init

net = nn.Sequential(
    nn.Linear(4, 3),
    nn.ReLU(),
    nn.Linear(3, 1)
)

print(net)

X = torch.rand(2, 4)
Y = net(X).sum()

# 初始化模型参数
print("1:", type(net.named_parameters()))
for name, param in net.named_parameters():
    print(name, param.size())
    if 'weight' in name:
        init.normal_(param, mean=0, std=0.01)
        print(name, param.data)
    if 'bias' in name:
        init.constant_(param, val=0)
        print(name, param.data)

print('****************************************************************')
# 权值共享
linear = nn.Linear(1, 1, bias=False)
net = nn.Sequential(linear, linear) 
print(net)
for name, param in net.named_parameters():
    init.constant_(param, val=3)
    print(name, param.data)
print(id(net[0]) == id(net[1]))
print(id(net[0].weight) == id(net[1].weight))