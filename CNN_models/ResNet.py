import time

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn
from torch import Tensor
from torch.utils import data


class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, stride=strides, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        y = torch.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return torch.relu(y)


class ResidualBlock(nn.Module):
    def __init__(self, input_channels, num_channels, num_residuals,
                 first_block=False):
        super(ResidualBlock, self).__init__()
        self.blk = nn.ModuleList()
        for i in range(num_residuals):
            if i == 0 and not first_block:
                self.blk.append(Residual(input_channels, num_channels,
                                         use_1x1conv=True, strides=2))
            else:
                self.blk.append(Residual(num_channels, num_channels))

    def forward(self, x):
        for b in self.blk:
            x = b(x)
        return x


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                                nn.BatchNorm2d(64), nn.ReLU(),
                                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                                )
        self.b2 = ResidualBlock(64, 64, 2, first_block=True)
        self.b3 = ResidualBlock(64, 128, 2)
        self.b4 = ResidualBlock(128, 256, 2)
        self.b5 = ResidualBlock(256, 512, 2)
        self.b6 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                nn.Flatten(),
                                nn.Linear(512, 10)
                                )

    def forward(self, x):
        x = self.b3(self.b2(self.b1(x)))
        x = self.b6(self.b5(self.b4(x)))
        return x


# 获取训练集和测试集
def get_iter(batch_size, resize=None):
    trans = [torchvision.transforms.ToTensor()]
    if resize:
        trans.insert(0, torchvision.transforms.Resize(resize))
    trans = torchvision.transforms.Compose(trans)  # 将多个组件组合在一起
    train_mnist = torchvision.datasets.FashionMNIST(
        root='../data', train=True, transform=trans, download=True
    )
    test_mnist = torchvision.datasets.FashionMNIST(
        root='../data', train=False, transform=trans, download=True
    )
    train_iter = data.DataLoader(train_mnist, batch_size=batch_size, shuffle=True)
    test_iter = data.DataLoader(test_mnist, batch_size=batch_size, shuffle=False)
    return train_iter, test_iter


# 参数初始化
def init_params(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_normal_(m.weight)


if __name__ == "__main__":
    # 超参数
    batch_size = 256
    gpu_id = 5
    lr = 0.05
    num_epochs = 10
    # 获取数据
    train_iter, test_iter = get_iter(batch_size=batch_size)
    # 声明网络
    net = ResNet()
    # 获取gpu
    device = torch.device(f'cuda:{gpu_id}')
    # 将网络移到gpu
    net.to(device=device)
    # 初始化网络
    net.apply(init_params)
    # 定义优化器
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    # 定义损失
    loss = nn.CrossEntropyLoss()
    # 记录损失、训练精度和测试精度
    mloss, mtracc, mteacc = \
        torch.zeros(num_epochs), torch.zeros(num_epochs), torch.zeros(num_epochs)
    # 开始时间
    time_start = time.time()
    # 开始训练
    for epoch in range(num_epochs):
        count_train = [0.0] * 3
        net.train()
        for X, y in train_iter:
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                count_train[0] += l * y.numel()
                count_train[1] += float(torch.sum(torch.argmax(net(X), dim=1) == y))
                count_train[2] += y.numel()
            time_end = time.time()
            print(f"\repoch {epoch + 1} - training - {count_train[2]}/60000"
                  f" examples: train_loss={count_train[0] / count_train[2] :.3f} "
                  f"train_acc={count_train[1] / count_train[2] :.3f}", end="")
        net.eval()
        count_test = [0.0] * 2
        with torch.no_grad():
            for X, y in test_iter:
                if isinstance(X, list):
                    X = [x.to(device) for x in X]
                else:
                    X = X.to(device)
                y = y.to(device)
                count_test[0] += float(torch.sum(torch.argmax(net(X), dim=1) == y))
                count_test[1] += y.numel()
                time_end = time.time()
                print(f"\repoch {epoch + 1} - testing - {count_test[1]}/10000"
                      f" examples", end="")
        time_end = time.time()
        print(
            f"\repoch {epoch + 1}: train_loss={count_train[0] / count_train[2] :.3f} "
            f"train_acc={count_train[1] / count_train[2] :.3f} "
            f"test_acc={count_test[0] / count_test[1] :.3f} "
            f"time_cost={time_end - time_start :.1f}")
        # 记录
        mloss[epoch], mtracc[epoch], mteacc[epoch] = count_train[0] / count_train[2], \
                                                     count_train[1] / count_train[2], \
                                                     count_test[0] / count_test[1]
    # 保存网络
    state = {'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': num_epochs}
    # torch.save(state, "../net/ResNet")
    print(
        f"{num_epochs * (count_train[2] + count_test[1]) / (time_end - time_start) :.1f} examples/sec on cuda:{gpu_id}")
    # 绘制图像
    plt.figure()
    x = np.arange(1, num_epochs + 1)
    plt.plot(x, mloss.detach().numpy(), label='loss', color='b')
    plt.plot(x, mtracc.detach().numpy(), label='train_acc', color='k')
    plt.plot(x, mteacc.detach().numpy(), label='test_acc', color='g')
    plt.xlabel("epoch")
    plt.title('ResNet')
    plt.legend(loc='best')
    plt.show()
    plt.savefig('../result/ResNet.png')

