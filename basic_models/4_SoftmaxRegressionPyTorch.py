import torch
import torchvision.datasets
from torch.utils import data
from torch import nn
from d2l import torch as d2l

batch_size = 256

# 获取数据集
trans = torchvision.transforms.ToTensor()
train_mnist = torchvision.datasets.FashionMNIST(
    root='../data', train=True, transform=trans, download=True
)
test_mnist = torchvision.datasets.FashionMNIST(
    root='../data', train=False, transform=trans, download=True
)
iter_train = data.DataLoader(train_mnist, batch_size=batch_size, shuffle=True)
iter_test = data.DataLoader(test_mnist, batch_size=batch_size, shuffle=False)

# 搭建模型并初始化
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 10)
)
net[1].weight.data.normal_(0, 0.01)
net[1].bias.data.fill_(0)

# 定义损失函数
loss = nn.CrossEntropyLoss(reduction='none')

# 定义优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

# 训练
num_epochs = 10

for epoch in range(num_epochs):
    # 进入训练模式
    net.train()
    metric = d2l.Accumulator(3)
    for X, y in iter_train:
        y_hat = net(X)
        l = loss(y_hat, y)
        trainer.zero_grad()     # 清空梯度
        l.mean().backward()     # 梯度回退
        trainer.step()          # 更新
        metric.azdd(float(l.sum()), d2l.accuracy(y_hat, y), y.numel())
    # 进入评估模式
    net.eval()
    metric2 = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in iter_test:
            metric2.add(d2l.accuracy(net(X), y), y.numel())
    print(f"epoch {epoch}: train_loss={metric[0]/metric[2] :.3f} train_acc={metric[1]/metric[2] :.3f} test_acc={metric2[0]/metric2[1]}")







