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

# 方法1
plt.annotate(r"$y=2x+1$",
             xy=(x0, y0),   # 标注点的信息
             xycoords='data',   # 信息是点的坐标数据
             xytext=(+30, -30),     # 标注文本的信息
             textcoords='offset points',    # 标注文本的信息是关于偏移的
             fontsize=16,           # 字体大小
             arrowprops=dict(arrowstyle='->',
                             connectionstyle='arc3, rad=.2')    # 配置箭头
             )


# 方法2
plt.text(-3.7, 3, r"$ZJU\ GOD\ HYH\ NB\ PLUS$",
         fontdict={'size': 16, 'color': 'r'})
plt.set_t
plt.show()