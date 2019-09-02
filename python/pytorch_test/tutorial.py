"""
    一个简单回归问题，可视化
"""


import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

# unsqueeze是将原本的一维数据变成二维
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)      # x: data(tensor), shape=(100,1)
y = x.pow(2) + 0.2 * torch.rand(x.size())

x, y = Variable(x), Variable(y)






class Net(torch.nn.Module):
    # 需要的信息
    def __init__(self, n_features, n_hidden, n_output):
        super(Net, self).__init__()     # 继承模块torch.nn.Module
        self.hidden = torch.nn.Linear(n_features, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)



    # 前向传递过程，真正搭建网络的地方
    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.predict(x)
        return x


net = Net(1, 10, 1)
print(net)

plt.ion()   # something about plotting,变成一个实时打印的过程
plt.show()

# 优化器
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)   # lr: learn rate
loss_func = torch.nn.MSELoss()      # 均方差处理，回归问题使用

for t in range(200):
    prediction = net(x)
    loss = loss_func(prediction, y)

    # 首先将梯度设置为0
    optimizer.zero_grad()
    loss.backward()
    # 优化梯度，学习效率为0.2
    optimizer.step()

    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()