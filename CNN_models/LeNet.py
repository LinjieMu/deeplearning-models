import numpy as np
import torch
import torchvision.transforms
from matplotlib import pyplot as plt
from torch import nn
from torch.utils import data
import time


# 定义模型
class LeNet(nn.Module):
    # 模型初始化
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.ft = nn.Flatten()
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.avg = nn.AvgPool2d(kernel_size=2, stride=2)

    # 前项传递函数
    def forward(self, x):
        x = self.avg(torch.sigmoid(self.conv1(x)))
        x = self.avg(torch.sigmoid(self.conv2(x)))
        x = self.ft(x)
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x


# 获取训练集和测试集
def get_iter(batch_size):
    trans = torchvision.transforms.ToTensor()
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
    batch_size = 512
    gpu_id = 7
    lr = 0.9
    num_epochs = 10
    # 获取数据
    train_iter, test_iter = get_iter(batch_size=batch_size)
    # 声明网络
    net = LeNet()
    # 获取gpu
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.device_count() > gpu_id else "cpu")
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
    # torch.save(state, "../net/LeNet")
    print(
        f"{num_epochs * (count_train[2] + count_test[1]) / (time_end - time_start) :.1f} examples/sec on cuda:{gpu_id}")
    # 绘制图像
    plt.figure()
    x = np.arange(1, num_epochs + 1)
    plt.plot(x, mloss.detach().numpy(), label='loss', color='b')
    plt.plot(x, mtracc.detach().numpy(), label='train_acc', color='k')
    plt.plot(x, mteacc.detach().numpy(), label='test_acc', color='g')
    plt.xlabel("epoch")
    plt.title('LeNet')
    plt.legend(loc='best')
    plt.show()
    # plt.savefig('../result/LeNet.png')

