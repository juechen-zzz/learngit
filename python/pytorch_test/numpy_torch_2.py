import torch
from torch.autograd import Variable

tensor = torch.FloatTensor([[1, 2], [3, 4]])
variable = Variable(tensor, requires_grad=True)  # 确定这个变量是否可以反向传播，会追踪所有对于该张量的操作
print(tensor)
print(variable)

t_out = torch.mean(tensor * tensor)
v_out = torch.mean(variable * variable)
print(t_out)
print(v_out)

v_out.backward()  # 反向传递, v_out = 1/4 * sum(var*var)
print('1', variable.grad)
print('2', variable.data)  # data是tensor形式，可以.numpy()
