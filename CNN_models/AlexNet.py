import torch
from torch import nn
import torchvision.transforms
from torch.utils import data
import time


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4, padding=1)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(6400, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)
        self.mp = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        x = self.mp(torch.relu(self.conv1(x)))
        x = self.mp(torch.relu(self.conv2(x)))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = self.mp(x)
        x = torch.flatten(x, 1)
        x = torch.dropout(torch.relu(self.fc1(x)), p=0.5, train=self.training)
        x = torch.dropout(torch.relu(self.fc2(x)), p=0.5, train=self.training)
        x = self.fc3(x)
        return x


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
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.device_count() > 7 else "cpu")
    # 定义神经网络，将其移到GPU中，并初始化参数
    net = AlexNet()
    net.to(device)
    net.apply(init_params())
    # 定义损失
    loss = nn.CrossEntropyLoss()
    # 定义优化器
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    # 开始时间
    time_start = time.time()
    # 开始训练
    for epoch in range(num_epochs):
        count_train = [0.0]*3
        net.train()
        for X, y in train_iter:
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                count_train[0] += l*y.numel()
                count_train[1] += float(torch.sum(torch.argmax(net(X), dim=1) == y))
                count_train[2] += y.numel()
            time_end = time.time()
            print(f"epoch {epoch+1} - {count_train[2]}/{len(train_iter)*batch_size}examples: train_loss={count_train[0] / count_train[2] :.3f} train_acc={count_train[1]/count_train[2] :.3f}")
        print(f"epoch {epoch+1}: train_loss={count_train[0]/count_train[2] :.3f} train_acc={count_train[1]/count_train[2] :.3f}",end=" ")
        net.eval()
        count_test = [0.0]*2
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
                print(f"epoch {epoch + 1} - {count_test[1]}/{len(test_iter)*batch_size}examples: test_acc={count_test[0]/count_test[1] :.3f} time_cost={time_end-time_start :.1f}")
        time_end = time.time()
        print(f"test_acc={count_test[0]/count_test[1] :.3f} time_cost={time_end-time_start :.1f}")
    #保存网络
    state = {'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': num_epochs}
    torch.save(state, "../net/AlexNet")
    print(f"{num_epochs*(count_train[2]+count_test[1])/(time_end-time_start) :.1f} examples/sec on cuda:{gpu_id}")

