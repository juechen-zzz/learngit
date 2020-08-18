import torch
from torch import nn

"""
torch.cuda.is_available() # 输出 True
torch.cuda.device_count() # 输出 4
torch.cuda.current_device() # 输出 0
torch.cuda.get_device_name(0) # 输出 'GeForce GTX 2080 Ti'
"""

x = torch.tensor([1, 2, 3])
x = x.cuda(0)                       # tensor([1, 2, 3], device='cuda:0')

x.device                            # device(type='cuda', index=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x = torch.tensor([1, 2, 3], device=device)
# or
x = torch.tensor([1, 2, 3]).to(device)
