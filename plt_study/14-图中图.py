import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
x = list(range(1, 8))
y = [1, 3, 4, 2, 5, 6, 8]

left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
ax1 = fig.add_axes([left, bottom, width, height])
ax1.plot(x, y, 'r')
ax1.set_title("F1")


left, bottom, width, height = 0.2, 0.6, 0.25, 0.25
ax2 = fig.add_axes([left, bottom, width, height])
ax2.plot(y, x, 'r')
ax2.set_title("F2")


plt.axes([.6, .2, .25, .25])
plt.plot(y[::-1], x, 'g')
plt.show()
