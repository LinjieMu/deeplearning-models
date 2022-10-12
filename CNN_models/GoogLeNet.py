import time

import numpy as np
import torch
import torchvision.transforms
from matplotlib import pyplot as plt
from torch.utils import data
from torch import nn


class Inception(nn.Module):
    # c1--c4是每条路径的输出通道数
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # 线路1，单1x1卷积层
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 线路2，1x1卷积层后接3x3卷积层
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3，1x1卷积层后接5x5卷积层
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路4，3x3最大汇聚层后接1x1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = torch.relu(self.p1_1(x))
        p2 = torch.relu(self.p2_2(torch.relu(self.p2_1(x))))
        p3 = torch.relu(self.p3_2(torch.relu(self.p3_1(x))))
        p4 = torch.relu(self.p4_2(self.p4_1(x)))
        # 在通道维度上连结输出
        return torch.cat((p1, p2, p3, p4), dim=1)


class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()
        self.b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                                )
        self.b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                                nn.ReLU(),
                                nn.Conv2d(64, 192, kernel_size=3, padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                                Inception(256, 128, (128, 192), (32, 96), 64),
                                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                                Inception(512, 160, (112, 224), (24, 64), 64),
                                Inception(512, 128, (128, 256), (24, 64), 64),
                                Inception(512, 112, (144, 288), (32, 64), 64),
                                Inception(528, 256, (160, 320), (32, 128), 128),
                                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                                Inception(832, 384, (192, 384), (48, 128), 128),
                                nn.AdaptiveAvgPool2d((1, 1)),
                                nn.Flatten(),
                                nn.Linear(1024, 10)
                                )

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        return x


# 获取训练集和测试集
def get_iter(batch_size, resize=None):
    trans = [torchvision.transforms.ToTensor()]
    if resize:
        trans.insert(0, torchvision.transforms.Resize(resize))
    trans = torchvision.transforms.Compose(trans)
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


if __name__ == '__main__':
    # 超参数
    batch_size = 128
    gpu_id = 5
    num_epochs = 10
    lr = 0.1
    # 获取数据集
    train_iter, test_iter = get_iter(batch_size=batch_size, resize=96)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.device_count() > gpu_id else "cpu")
    # 定义网络
    net = GoogLeNet()
    print(net)
    net.to(device)
    net.apply(init_params)
    # 定义损失
    loss = nn.CrossEntropyLoss()
    # 定义优化器
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
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
    # state = {'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': num_epochs}
    # torch.save(state, "../net/GoogLeNet")
    print(
        f"{num_epochs * (count_train[2] + count_test[1]) / (time_end - time_start) :.1f} examples/sec on cuda:{gpu_id}")
    # 绘制图像
    plt.figure()
    x = np.arange(1, num_epochs + 1)
    plt.plot(x, mloss.detach().numpy(), label='loss', color='b')
    plt.plot(x, mtracc.detach().numpy(), label='train_acc', color='k')
    plt.plot(x, mteacc.detach().numpy(), label='test_acc', color='g')
    plt.xlabel("epoch")
    plt.title('GoogLeNet')
    plt.legend(loc='best')
    plt.show()
    # plt.savefig('../result/GoogLeNet.png')


