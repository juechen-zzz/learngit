# 自动求梯度

import torch

# 每个Tensor都有一个.grad_fn属性，该属性即创建该Tensor的Function, 就是说该Tensor是不是通过某些运算得到的
# 若是，则grad_fn返回一个与这些运算相关的对象，否则是None。
# 此外，如果我们想要修改tensor的数值，但是又不希望被autograd记录（即不会影响反向传播），那么我么可以对tensor.data进行操作。

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

out.backward()      # 等价于 out.backward(torch.tensor(1.))
print(x.grad)

print("--------------------------------------------")
x1 = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
y1 = 2 * x1
z1 = y1.view(2, 2)
print(x1.grad)

v = torch.tensor([[1.0, 0.1], [0.01, 0.001]], dtype=torch.float)
z1.backward(v)
print(x1.grad)


# with torch.no_grad():
#     y2 = x ** 3
