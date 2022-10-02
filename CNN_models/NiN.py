import torch
import time
import torchvision
from torch import nn
from torch.utils import data


# 定义一个NiN块：NiN块有1个卷积层和两个1*1卷积层构成
def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU()
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
    batch_size = 256
    gpu_id = 7
    num_epochs = 10
    lr = 0.1
    # 获取数据集
    train_iter, test_iter = get_iter(batch_size=batch_size, resize=224)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.device_count() > gpu_id else "cpu")
    # 定义网络
    net = nn.Sequential(
        nin_block(1, 96, kernel_size=11, strides=4, padding=0),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nin_block(96, 256, kernel_size=5, strides=1, padding=2),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nin_block(256, 384, kernel_size=3, strides=1, padding=1),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Dropout(0.5),
        # 标签类别数为10
        nin_block(384, 10, kernel_size=3, strides=1, padding=1),
        nn.AdaptiveAvgPool2d((1, 1)),
        # 将四维张量转化为二维输出
        nn.Flatten()
    )
    print(net)
    X = torch.rand(size=(1, 1, 224, 224))
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shapes:\t', X.shape)
    net.apply(init_params)
    net.to(device)
    # 定义损失
    loss = nn.CrossEntropyLoss()
    # 定义优化器
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    # 开始时间
    time_start = time.time()
    # 开始训练
    print("=" * 6 + "Train Start" + "=" * 6 + "\n")
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
            print(f"epoch {epoch + 1} - {count_train[2]}/{len(train_iter) * batch_size}: "
                  f"train_loss={count_train[0] / count_train[2] :.3f} "
                  f"train_acc={count_train[1] / count_train[2] :.3f}")
        print(f"epoch {epoch + 1}: train_loss={count_train[0] / count_train[2] :.3f} "
                   f"train_acc={count_train[1] / count_train[2] :.3f} ")
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
                print(f"epoch {epoch + 1} - {count_test[1]}/{len(test_iter) * batch_size}: "
                      f"test_acc={count_test[0] / count_test[1] :.3f} ")
        time_end = time.time()
        print(f"test_acc={count_test[0] / count_test[1] :.3f} "
                   f"time_cost={time_end - time_start :.1f}\n")
    # 保存网络
    state = {'net': net.state_dict(),
             'optimizer': optimizer.state_dict(),
             'epoch': num_epochs}
    torch.save(state, "../net/NiN")
    print(f"{num_epochs * (count_train[2] + count_test[1]) / (time_end - time_start) :.1f}"
          f" examples/sec on cuda:{gpu_id}\n")
