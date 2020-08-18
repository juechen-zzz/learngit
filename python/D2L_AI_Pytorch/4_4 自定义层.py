import torch 
from torch import nn

class CenteredLayer(nn.Module):
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)
    
    def forward(self, x):
        return x - x.mean()
    
layer = CenteredLayer()
print(layer(torch.tensor([1, 2, 3, 4, 5], dtype=torch.float)))

# 构造更复杂的模型
net = nn.Sequential(
    nn.Linear(8, 128),
    CenteredLayer()
)

y = net(torch.rand(4, 8))
print(y.mean().item())


class MyDense(nn.Module):
    def __init__(self):
        super(MyDense, self).__init__()
        # ParameterList接收一个Parameter实例的列表作为输入然后得到一个参数列表，使用的时候可以用索引来访问某个参数，
        # 另外也可以使用append和extend在列表后面新增参数。
        self.params = nn.ParameterList([nn.Parameter(torch.randn(4, 4)) for i in range(3)])
        self.params.append(nn.Parameter(torch.randn(4, 1)))
    
    def forward(self, x):
        for i in range(len(self.params)):
            x = torch.mm(x, self.params[i])
        return x

net = MyDense()
print(net)


class MyDictDense(nn.Module):
    def __init__(self):
        super(MyDictDense, self).__init__()
        # ParameterDict接收一个Parameter实例的字典作为输入然后得到一个参数字典，然后可以按照字典的规则使用了。
        # 例如使用update()新增参数，使用keys()返回所有键值，使用items()返回所有键值对等
        self.params = nn.ParameterDict({
            'linear1': nn.Parameter(torch.randn(4, 4)),
            'linear2': nn.Parameter(torch.randn(4, 1))
        })
        self.params.update({'linear3': nn.Parameter(torch.randn(4, 2))})
    
    def forward(self, x, choice='linear1'):
        return torch.mm(x, self.params[choice])

net = MyDictDense()
print(net)



net = nn.Sequential(
    MyDictDense(),
    MyDense()
)
print(net)
x = torch.ones(1, 4)
print(net(x))