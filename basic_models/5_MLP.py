import torch
import torchvision
from torch import nn
from torch.utils import data
from d2l import torch as d2l


# 获取数据集
batch_size = 256
trans = torchvision.transforms.ToTensor()
train_mnist = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=trans, download=True
)
test_mnist = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True
)
train_iter = data.DataLoader(train_mnist, batch_size=batch_size, shuffle=True)
test_iter = data.DataLoader(test_mnist, batch_size=batch_size, shuffle=False)


# 初始化模型参数
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)


def init_weights(m):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.fill_(0)


net.apply(init_weights)
# 定义损失函数
loss = nn.CrossEntropyLoss(reduction='none')
# 定义优化器
lr = 0.1
optimizor = torch.optim.SGD(net.parameters(),lr=lr)

# 训练
num_epochs = 20
for epoch in range(num_epochs):
    # 训练模式
    net.train()
    metric1 = d2l.Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        optimizor.zero_grad()   # 清空梯度
        l.mean().backward()     # 梯度回退
        optimizor.step()        # 更新
        metric1.add(l.sum(), d2l.accuracy(y_hat, y), y.numel())
    # 测试模式
    metric2 = d2l.Accumulator(2)
    for X, y in test_iter:
        metric2.add(d2l.accuracy(net(X), y), y.numel())
    print(f"epoch {epoch}: train_loss={metric1[0]/metric1[2] :.3f} "
          f"train_acc={metric1[1]/metric1[2] :.3f} "
          f"test_acc={metric2[0]/metric2[1] :.3f}")



