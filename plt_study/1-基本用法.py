import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-1, 1, 50)
# y = 2 * x + 1
y = x ** 2
plt.plot(x, y)
# 在脚本中只有调用了show方法才会有图片显示
plt.show()
