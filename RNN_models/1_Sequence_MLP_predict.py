import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils import data

# 生成1000个样本点
T = 1000
time = torch.tensor(range(1, T + 1), dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, time.shape)

# 绘图
plt.figure()
plt.plot(time.detach().numpy(), x.detach().numpy(), color='blue', label='True Value')

tau = 4
features = torch.zeros((T - tau, tau))  # 前tau个样本无法被预测
for i in range(tau):
    features[:, i] = x[i:T - tau + i]
labels = x[tau:].reshape((-1, 1))

# 封装数据集
batch_size, n_train = 16, 600
train_dataset = data.TensorDataset(*(features[:n_train], labels[:n_train]))
test_dataset = data.TensorDataset(*(features[n_train:], labels[n_train:]))
# 生成迭代器
train_iter = data.DataLoader(train_dataset, batch_size, shuffle=True)
test_iter = data.DataLoader(test_dataset, batch_size, shuffle=False)

# 构造神经网络
net = nn.Sequential(nn.Linear(tau, 10),
                    nn.ReLU(),
                    nn.Linear(10, 1))


# 初始化网络
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


net.apply(init_weights)

# 平方损失
loss = nn.MSELoss(reduction='none')
# 优化器
lr = 0.01
trainer = torch.optim.Adam(net.parameters(), lr=lr)
num_epochs = 4
for epoch in range(num_epochs):
    loss_ = 0
    for X, y in train_iter:
        trainer.zero_grad()
        y_hat = net(X)
        l = loss(y_hat, y)
        l.sum().backward()
        trainer.step()
        loss_ += l.sum()
    print(f'epoch {epoch + 1}, '
          f'loss: {loss_ / n_train:f}', end=" ")
    with torch.no_grad():
        loss_ = 0
        for X, y in test_iter:
            y_hat = net(X)
            l = loss(y_hat, y)
            loss_ += l.sum()
    print(f'test_loss: {loss_ / (T - n_train):f}')

# 预测
y_hat = net(features)
plt.plot(time[tau:].detach().numpy(), y_hat.detach().numpy(), color='red', alpha=0.5, label='Predict-1')
N = T - n_train
time_p = torch.tensor(range(n_train, T), dtype=torch.float32)
x_p = torch.zeros((1, 4))
x_p = x[n_train - tau:n_train]
y_p = torch.zeros_like(time_p)
for i in range(n_train, T):
    y_p[i - n_train] = net(x_p)
    x_p[:-1] = x_p[1:].clone()
    x_p[-1] = y_p[i - n_train].clone()
plt.plot(time_p.detach().numpy(), y_p.detach().numpy(), color='green', alpha=0.5, label='Predict-2')
plt.legend(loc='best')
plt.title(f"step={tau} result")
plt.show()
