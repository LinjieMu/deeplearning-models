import time

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn
from torch.utils import data


# 定义一个改良版的“批量规范化、激活和卷积”架构
def cov_block(in_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(in_channels), nn.ReLU(),
        nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1)
    )


# 一个稠密块由多个卷积块组成，每个卷积块使用相同数量的输出通道。
class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(cov_block(
                num_channels * i + input_channels, num_channels
            ))
        self.net = nn.Sequential(*layer)

    def forward(self, x):
        for blk in self.net:
            y = blk(x)
            x = torch.cat((x, y), dim=1)
        return x


# 过渡层通过1x1卷积层来减小通道数，并使用步幅为2的平均汇聚层减半高和宽
def transition_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2))


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


if __name__ == '__main__':
    # 定义网络
    b1 = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    # num_channels为当前的通道数
    num_channels, growth_rate = 64, 32
    num_convs_in_dense_blocks = [4, 4, 4, 4]
    blks = []
    for i, num_convs in enumerate(num_convs_in_dense_blocks):
        blks.append(DenseBlock(num_convs, num_channels, growth_rate))
        # 上一个稠密块的输出通道数
        num_channels += num_convs * growth_rate
        # 在稠密块之间添加一个转换层，使通道数量减半
        if i != len(num_convs_in_dense_blocks) - 1:
            blks.append(transition_block(num_channels, num_channels // 2))
            num_channels = num_channels // 2
    net = nn.Sequential(
        b1, *blks,
        nn.BatchNorm2d(num_channels), nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(num_channels, 10))
    # 超参数
    batch_size = 256
    gpu_id = 6
    lr = 0.1
    num_epochs = 10
    # 获取数据
    train_iter, test_iter = get_iter(batch_size=batch_size, resize=96)
    # 获取gpu
    device = torch.device(f"cuda:{gpu_id}"
                          if torch.cuda.device_count() > gpu_id else "cpu")
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
    # state = {'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': num_epochs}
    # torch.save(state, "../net/DenseNet")
    print(
        f"{num_epochs * (count_train[2] + count_test[1]) / (time_end - time_start) :.1f} examples/sec on cuda:{gpu_id}")
    # 绘制图像
    plt.figure()
    x = np.arange(1, num_epochs + 1)
    plt.plot(x, mloss.detach().numpy(), label='loss', color='b')
    plt.plot(x, mtracc.detach().numpy(), label='train_acc', color='k')
    plt.plot(x, mteacc.detach().numpy(), label='test_acc', color='g')
    plt.xlabel("epoch")
    plt.title('DenseNet')
    plt.legend(loc='best')
    plt.show()
    # plt.savefig('../result/DenseNet.png')


