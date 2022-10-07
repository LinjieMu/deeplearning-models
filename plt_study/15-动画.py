import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

fig, ax = plt.subplots()
x = np.arange(0, 2*np.pi, 0.01)
line, = ax.plot(x, np.sin(x))


def animate(i):
    line.set_ydata(np.sin(x+i/20))
    return line,


def init():
    line.set_ydata(np.sin(x))
    return line,


ani = animation.FuncAnimation(fig=fig,
                              func=animate,
                              frames=100,   # 更新多少帧
                              init_func=init,
                              interval=20,  # 更新的频率
                              blit=False
                              )
plt.show()