import torch

net = torch.nn.Linear(10, 1).cuda()
print(net)      # Linear(in_features=10, out_features=1, bias=True)

net = torch.nn.DataParallel(net, device_ids=[0, 3])

torch.save(net.state_dict(), "./8.4_model.pt")
new_net = torch.nn.Linear(10, 1)
new_net = torch.nn.DataParallel(new_net)
new_net.load_state_dict(torch.load("./8.4_model.pt")) # 加载成功