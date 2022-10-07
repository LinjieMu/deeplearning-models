"""
    本节介绍关于坐标轴设置的一些代码
    xlim方法：设置x坐标轴的范围
    ylim方法：设置y坐标轴的范围
    xlabel方法：设置x坐标轴标签名
    ylabel方法：设置y坐标轴标签名
    xticks方法：设置x轴的刻度数值
    yticks方法：设置y轴的刻度数值

    gca方法： get current axis 获取当前图像的坐标对象
        - spines属性，图像的四个脊梁
            - set_color方法：为属性设置相应的颜色
            - set_position方法：将当前轴移到特定的位置
        - xaxis属性，图像的x轴;yaxis属性，图像的y轴
            - set_ticks_position方法：设置该轴为某个位置

"""
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3, 3, 50)
y1 = 2 * x + 1
y2 = x ** 2
# 绘制图片
plt.figure(num=3, figsize=(8, 5))
plt.plot(x, y2)
plt.plot(x, y1, color='red', linewidth=1, linestyle='--')

# 设置坐标轴的范围
plt.xlim((-1, 2))
plt.ylim((-2, 3))

# 设置坐标轴标签名
plt.xlabel("I am x")
plt.ylabel("I am y")

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

# 设置脊梁
ax = plt.gca()
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.xaxis.set_ticks_position('bottom')   # 设置x轴为下面的脊梁
ax.yaxis.set_ticks_position('left')   # 设置y轴为左面的脊梁
ax.spines['bottom'].set_position(('data', 0))    # 将x轴移到y轴对应的0点
ax.spines['left'].set_position(('data', 0))      # 将y轴移到x轴对应的0点
# 显示图片
plt.show()
