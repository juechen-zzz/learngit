import torch
from torch import nn

# method 1
class MLP(nn.Module):
    # 声明带有模型参数的层，这里声明2个全连接层
    def __init__(self, **kwargs):
        # 调用MLP父类的构造函数来进行必要的初始化，这样在构造实例时还可以指定其他函数
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Linear(784, 256)
        self.act = nn.ReLU()
        self.output = nn.Linear(256, 10)
    
    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)

# method 2
# net = MySequential(
#         nn.Linear(784, 256),
#         nn.ReLU(),
#         nn.Linear(256, 10), 
#         )

# method 3
# net = nn.ModuleList([nn.Linear(784, 256), nn.ReLU()])
# net.append(nn.Linear(256, 10)) # # 类似List的append操作

# ModuleList仅仅是一个储存各种模块的列表，这些模块之间没有联系也没有顺序（所以不用保证相邻层的输入输出维度匹配），
# 而且没有实现forward功能需要自己实现，所以上面执行net(torch.zeros(1, 784))会报NotImplementedError；
# 而Sequential内的模块需要按照顺序排列，要保证相邻层的输入输出大小相匹配，内部forward功能已经实现。
# 加入到ModuleList里面的所有模块的参数会被自动添加到整个网络中

X = torch.rand(2, 784)
net = MLP()
print(net)
