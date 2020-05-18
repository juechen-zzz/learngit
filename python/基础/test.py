# 自动求梯度
# 如果有一个函数值和自变量都为向量的函数,梯度就是一个雅可比矩阵（Jacobian matrix）
# torch.autograd这个包就是用来计算一些雅克比矩阵的乘积的

import torch

# 每个Tensor都有一个.grad_fn属性，该属性即创建该Tensor的Function, 就是说该Tensor是不是通过某些运算得到的
# 若是，则grad_fn返回一个与这些运算相关的对象，否则是None。
print("x--------------------------")
x = torch.ones(2, 2, requires_grad=True)
print(x)
print(x.grad_fn)

print("y--------------------------")
y = x + 2
print(y)
print(y.grad_fn)

print("z--------------------------")
z = y * y * 3
out = z.mean()
print(z)
print(out)

out.backward() # 等价于 out.backward(torch.tensor(1.))
print(x.grad)