import torch
from torch import nn

# x = torch.ones(3)
# print(x)
# torch.save(x, 'x.pth')
# x2 = torch.load('x.pth')
# print(x2)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(3, 2)
        self.act = nn.ReLU()
        self.output = nn.Linear(2, 1)
    
    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)

net = MLP()
print(net.state_dict())    # state_dict是一个从参数名称隐射到参数Tesnor的字典对

optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
print(optimizer.state_dict())

torch.save(net.state_dict(), 'net.pth')

model = MLP()
model.load_state_dict(torch.load('net.pth'))
print(model)