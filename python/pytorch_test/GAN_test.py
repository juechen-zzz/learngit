"""
    生成对抗网络
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# Hyper parameters
BATCH_SIZE = 64
LR_G = 0.0001           # 生成器的学习率
LR_D = 0.0001           # 判别器的学习率
N_IDEAS = 5             # 生成一个作品的想法数目，可以理解为输入的点
ART_COMPOENTS = 15      # 最终输出的点，或者说是最后的输出点
PAINT_POINTS = np.vstack([np.linspace(-1, 1, ART_COMPOENTS) for _ in range(BATCH_SIZE)])

# plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
# plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
# plt.legend(loc='upper right')
# plt.show()

def artist_work():      # 从真实目标中学习
    a = np.random.uniform(1, 2, size=BATCH_SIZE)[:, np.newaxis]
    paintings = a * np.power(PAINT_POINTS, 2) + (a - 1)
    paintings = torch.from_numpy(paintings).float()
    return paintings

G = nn.Sequential(
    nn.Linear(N_IDEAS, 128),
    nn.ReLU(),
    nn.Linear(128, ART_COMPOENTS),
)

D = nn.Sequential(
    nn.Linear(ART_COMPOENTS, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid(),
)

opt_D = torch.optim.Adam(D.parameters(), lr=LR_D)
opt_G = torch.optim.Adam(G.parameters(), lr=LR_G)

plt.ion()

for step in range(10000):
    artist_paintings = artist_work()                # 真实值
    G_ideas = torch.randn(BATCH_SIZE, N_IDEAS)      # 随机想法
    G_paintings = G(G_ideas)                        # 生成器的输出

    prob_artist0 = D(artist_paintings)              # 判别器尝试增加这项概率，使结果更像真的
    prob_artist1 = D(G_paintings)                   # 判别器尝试减少这项概率

    D_loss = - torch.mean(torch.log(prob_artist0) + torch.log(1. - prob_artist1))
    G_loss = torch.mean(torch.log(1. - prob_artist1))

    opt_D.zero_grad()
    D_loss.backward(retain_graph=True)
    opt_D.step()

    opt_G.zero_grad()
    G_loss.backward(retain_graph=True)
    opt_G.step()

    if step % 50 == 0:  # plotting
        plt.cla()
        plt.plot(PAINT_POINTS[0], G_paintings.data.numpy()[0], c='#4AD631', lw=3, label='Generated painting', )
        plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
        plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
        plt.text(-.5, 2.3, 'D accuracy=%.2f (0.5 for D to converge)' % prob_artist0.data.numpy().mean(),
                 fontdict={'size': 13})
        plt.text(-.5, 2, 'D score= %.2f (-1.38 for G to converge)' % -D_loss.data.numpy(), fontdict={'size': 13})
        plt.ylim((0, 3))
        plt.legend(loc='upper right', fontsize=10)
        plt.draw()
        plt.pause(0.01)

plt.ioff()
plt.show()