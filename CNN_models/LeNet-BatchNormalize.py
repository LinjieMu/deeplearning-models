import torch
import torchvision.transforms
from torch import nn
from torch.utils import data
import time


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
    batch_size = 256
    gpu_id = 6
    lr = 0.9
    epoch_nums = 10
    # 获取数据
    train_iter, test_iter = get_iter(batch_size=batch_size)
    # 声明网络
    net = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5), nn.BatchNorm2d(6), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(256, 120), nn.BatchNorm1d(120), nn.Sigmoid(),
        nn.Linear(120, 84), nn.BatchNorm1d(84), nn.Sigmoid(),
        nn.Linear(84, 10)
    )
    # 获取gpu
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.device_count() > gpu_id else "cpu")
    # 将网络移到gpu
    net.to(device=device)
    # 初始化网络
    net.apply(init_params)
    # 定义优化器
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    # 定义损失
    loss = nn.CrossEntropyLoss()
    # 定义开始时间
    time_start = time.time()
    # 训练
    for epoch in range(epoch_nums):
        # 训练损失之和，训练准确率之和，样本数
        count = [0.0, 0.0, 0.0]
        net.train()  # 进入训练模式
        for X, y in train_iter:
            trainer.zero_grad()  # 梯度清零
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            trainer.step()
            with torch.no_grad():
                count[0] += l * y.numel()
                count[1] += float(torch.sum(torch.argmax(net(X), dim=1) == y))
                count[2] += y.numel()
        print(f"epoch {epoch + 1}: train_loss={count[0] / count[2] :0.3f}, train_acc={count[1] / count[2] :0.3f}",
              end=" ")
        net.eval()  # 设置为评估模式
        # 正确预测数，预测总数
        test_count = [0.0, 0.0]
        with torch.no_grad():
            for X, y in test_iter:
                if isinstance(X, list):
                    X = [x.to(device) for x in X]
                else:
                    X = X.to(device)
                y = y.to(device)
                test_count[0] += float(torch.sum(torch.argmax(net(X), dim=1) == y))
                test_count[1] += float(y.numel())
        time_end = time.time()
        print(f"test_acc={test_count[0] / test_count[1] :0.3f} time_cost={time_end - time_start} :.2f")
        # 保存网络
    state = {'net': net.state_dict(), 'optimizer': trainer.state_dict(), 'epoch': epoch_nums}
    torch.save(state, '../net/LeNet-BN')
    print(f"{epoch_nums * (test_count[1] + count[2]) / (time_end - time_start) :.1f} examples/sec on cuda:{gpu_id}")
