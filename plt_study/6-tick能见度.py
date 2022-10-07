import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3, 3, 50)
y = 2*x + 1

plt.figure(num=1, figsize=(8, 5))
plt.plot(x, y)

ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['left'].set_position(('data', 0))
ax.spines['bottom'].set_position(('data', 0))

# 下面示意一个点和虚线的绘制方式
x0 = 1
y0 = 2*x0 + 1
plt.scatter(x0, y0, color='b')   # 打散点的方法，与plot打直接的方法类比
plt.plot([x0, x0], [0, y0], 'k--', lw=3)

for label in ax.get_xticklabels()+ax.get_yticklabels():
    label.set_fontsize(12)      # 设置刻度标签字体
    label.set_bbox(dict(facecolor='white',  # 设置刻度标签背景色
                        edgecolor='none',   # 设置刻度边框色
                        alpha=0.7))  # 设置透明度

plt.show()
