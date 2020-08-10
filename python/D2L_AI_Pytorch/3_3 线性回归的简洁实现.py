import torch
import numpy as np
import torch.utils.data as Data
import torch.nn as nn
from torch.nn import init
import torch.optim as optim

num_inputs = 2                  # 两个特征
num_examples = 1000             # 每个特征有1000个样例
true_w = [2, -3.4]
true_b = 4.2
features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float32)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

# 读取数据 PyTorch提供了data包来读取数据。由于data常用作变量名，我们将导入的data模块用Data代替。
# 在每一次迭代中，我们将随机读取包含10个数据样本的小批量。
batch_size = 10
dataset = Data.TensorDataset(features, labels)
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)

for X, y in data_iter:
    print(X, y)
    break

class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)

    def forward(self, x):
        y = self.linear(x)
        return y

net = LinearNet(num_inputs)
print(net)

init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)  # 也可以直接修改bias的data: net[0].bias.data.fill_(0)

# 定义损失函数
loss = nn.MSELoss()

optimizer = optim.SGD(net.parameters(), lr=0.03)
# print(optimizer)

num_epochs = 5
for epoch in range(1, num_epochs+1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))     # view(-1,1)指定了第二维个数为1，自动计算第一维
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss %f' % (epoch, l.item())) # 返回可遍历的(键, 值) 元组数组

dense = net.linear
print("true_w:", true_w)
print("true_b:", true_b)
print("dense.weight:", dense.weight)
print("dense.bias:", dense.bias)
