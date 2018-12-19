'''
@Description: pytorch玩耍
@Version: 
@Author: liguoying@iiotos.com
@Date: 2018-12-07 16:42:21
@LastEditTime: 2018-12-07 17:12:03
@LastEditors: Please set LastEditors
'''

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + 0.2 * torch.rand(x.size())

# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

net = Net(n_feature=1, n_hidden=10, n_output=1)

print("网络结构为：\n", net)

# 训练网络
optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
loss_func = torch.nn.MSELoss()

for t in range(2000):
    predict = net(x)
    loss = loss_func(predict, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 5 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), predict.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'loss=%.4f'%loss.data.numpy(),
                    fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)