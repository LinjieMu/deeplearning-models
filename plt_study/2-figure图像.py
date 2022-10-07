"""
    如果直接在matplotlib中绘制多张图，后面的图会和前面的绘制在一起，本节介绍
    的figure方法可以同时显示多张图片。
    figure方法:
        - num -> int 表示图片的编号
        - figsize -> tuple 表示图片大小
    plot方法：
        - color -> string 指定线条颜色
        - linewidth -> float 线宽
        - linestyle -> string 指定线条类型

"""
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3, 3, 50)
y1 = 2*x + 1
y2 = x**2
# 绘制图片
plt.figure()
plt.plot(x, y1)
# 绘制图片
plt.figure(num=3, figsize=(8, 5))
plt.plot(x, y2)
plt.plot(x, y1, color='red', linewidth=10, linestyle='--')
# 显示图片
plt.show()
