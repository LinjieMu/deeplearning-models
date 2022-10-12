import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils import data
import torchvision
import time


# 定义一个vgg块
def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


# 将多个vgg块组和为一个网络
def VGG(conv_arch):
    conv_blks = []
    in_channels = 1
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels
    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(p=0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(p=0.5),
        nn.Linear(4096, 10)
    )


# 获取训练集和测试集
def get_iter(batch_size, resize=None):
    trans = [torchvision.transforms.ToTensor()]
    if resize:
        trans.insert(0, torchvision.transforms.Resize(resize))
    trans = torchvision.transforms.Compose(trans)       # 将多个组件组合在一起
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
    batch_size = 512
    gpu_id = 6
    num_epochs = 10
    lr = 0.1
    # 获取数据集
    train_iter, test_iter = get_iter(batch_size=batch_size, resize=224)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.device_count() > gpu_id else "cpu")
    # 定义网路，初始化网络，将网络移到GPU中
    conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
    net = VGG(conv_arch)
    net.apply(init_params)
    net.to(device)
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
    # torch.save(state, "../net/VGG-11")
    print(
        f"{num_epochs * (count_train[2] + count_test[1]) / (time_end - time_start) :.1f} examples/sec on cuda:{gpu_id}")
    # 绘制图像
    plt.figure()
    x = np.arange(1, num_epochs + 1)
    plt.plot(x, mloss.detach().numpy(), label='loss', color='b')
    plt.plot(x, mtracc.detach().numpy(), label='train_acc', color='k')
    plt.plot(x, mteacc.detach().numpy(), label='test_acc', color='g')
    plt.xlabel("epoch")
    plt.title('VGG-11')
    plt.legend(loc='best')
    plt.show()
    # plt.savefig('../result/VGG-11.png')


