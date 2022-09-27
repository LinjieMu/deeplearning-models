import torch
import torchvision.transforms
from torch import nn
from torch.utils import data


# 定义模型
class LeNet(nn.Module):
    # 模型初始化
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.sg = nn.Sigmoid()
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
    batch_size = 256
    gpu_id = 2
    lr = 0.1
    epoch_nums = 10
    # 获取数据
    train_iter, test_iter = get_iter(batch_size=batch_size)
    # 声明网络
    net = LeNet()
    # 获取gpu
    device = torch.device(f'cuda:{gpu_id}')
    # 将网络移到gpu
    net.to(device=device)
    # 初始化网络
    net.apply(init_params)
    # 定义优化器
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    # 定义损失
    loss = nn.CrossEntropyLoss()
    # 训练
    for epoch in range(epoch_nums):
        # 训练损失之和，训练准确率之和，样本数
        count = [0.0, 0.0, 0.0]
        net.train()             # 进入训练模式
        for i, (X, y) in enumerate(train_iter):
            trainer.zero_grad()     # 梯度清零
            y = y.type(torch.float32)
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            trainer.step()
            with torch.no_grad():
                count[0] += l * y.numel()
                count[1] += float(torch.sum(torch.argmax(net(X)) == y))
                count[2] += y.numel()
            if i % 5 == 0:
                print(f"epoch {epoch} stage {i}: train_loss={count[0]/count[2]}, train_acc={count[1]/count[2]}")
        net.eval()  # 设置为评估模式
        # 正确预测数，预测总数
        test_count = [0.0, 0.0]
        with torch.no_grad():
            for X, y in test_iter:
                y = y.type(torch.float32)  # 调整y的数据类型与X一致
                if isinstance(X, list):
                    X = [x.to(device) for x in X]
                else:
                    X = X.to(device)
                y = y.to(device)
                test_count[0] += float(torch.sum(torch.argmax(net(X), dim=1) == y))
                test_count[1] += float(y.numel())
                print(f"epoch {epoch} test stage: test_acc={test_count[0] / test_count[1]}")
        # 保存网络
        torch.save(net.state_dict())

