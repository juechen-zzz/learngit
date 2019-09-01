"""
    numpy和torch对比
"""

import torch
import numpy as np

# 新建一个numpy数组，[[0,1,2],[3,4,5]]
np_data = np.arange(6).reshape((2, 3))
torch_data = torch.from_numpy(np_data)
tensor2array = torch_data.numpy()

print(
    '\nnumpy:', np_data,
    '\ntorch', torch_data,
    '\ntensor2array', tensor2array,
)


# abs
data = [[-1, -2], [1, 2]]
tensor = torch.FloatTensor(data)    # 32bit

print(
    '\nnumpy:', np.matmul(data, data),          # data.dot(data)     结果相同
    '\ntorch:', torch.mm(tensor, tensor)        # tensor.dot(tensor) 点乘相加
)
