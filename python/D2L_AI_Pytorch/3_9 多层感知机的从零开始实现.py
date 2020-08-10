import torch
import numpy as np
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_input = 784
num_output = 10
num_hidden = 256

w1 = torch.tensor(np.random.normal(0, 0.01, (num_input, num_hidden)), dtype=torch.float32)
b1 = torch.zeros(num_hidden, dtype=torch.float32)
w2 = torch.tensor(np.random.normal(0, 0.01, (num_hidden, num_output)), dtype=torch.float32)
b2 = torch.zeros(num_output, dtype=torch.float32)

params = [w1, b1, w2, b2]
for para in params:
    para.requires_grad_(requires_grad=True)

def relu(X):
    return torch.max(input=X, other=torch.tensor(0.0))

def net(X):
    X = X.view((-1, num_input))
    H = relu(torch.matmul(X, w1) + b1)
    return torch.matmul(H, w2) + b2

loss = torch.nn.CrossEntropyLoss()

num_epoch = 5
lr = 100.0

def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            if optimizer is None:
                d2l.sgd(params, lr, batch_size)
            else:
                optimizer.step()  # “softmax回归的简洁实现”一节将用到


            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = d2l.evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

train_ch3(net, train_iter, test_iter, loss, num_epoch, batch_size, params, lr)
