"""
    本节主要介绍如何制作图例Legend
    划线时加上label标签，在结束时调用legend()方法绘制图例。
    legend方法相关参数：
    - loc：显示图例的位置。
        upper center lower| right center left 的组合
        best 自动放到最合适的位置
    - handles 和 labels: 传入两个列表，一个含有线条对象，一个含有标签名。
        也可以在声明线条对象时传入label参数，二选一。
        另外，注意在等式左边接线条对象时注意后面有个逗号。

"""
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3, 3, 50)
y1 = 2 * x + 1
y2 = x ** 2
# 绘制图片
plt.figure(num=3, figsize=(8, 5))
l1, = plt.plot(x, y2, label='up')
l2, = plt.plot(x, y1, color='red', linewidth=1, linestyle='--', label='down')

# 设置坐标轴的范围
plt.xlim((-1, 2))
plt.ylim((-2, 3))

# 设置坐标轴的刻度数值
# 方法一：直接给出一个列表显示对应刻度
new_ticks = np.linspace(-1, 2, 5)
plt.xticks(new_ticks)
# 方法二：将刻度对应为字符串
# plt.yticks([-2, -1.8, -1, 1.22, 3],
#            ['really bad', 'bad', 'normal', 'good', 'really good'])
# 注：Matplotlib中原生支持数学公式，语法与markdown中相同
plt.yticks([-2, -1.8, -1, 1.22, 3],
           [r'$really\ bad$', r'$bad$', r'$normal$', r'$good$', r'$really\ good$'])


plt.legend(handles=[l1, l2], labels=["aaaa", "bbb"], loc="best")
# 显示图片
plt.show()
