import matplotlib.pyplot as plt
import numpy as np

n = 12
X = np.arange(n)
Y1 = (1 - X/float(n))*np.random.uniform(0.5, 1, n)
Y2 = (1 - X/float(n))*np.random.uniform(0.5, 1, n)


plt.bar(X, +Y1, facecolor='#9999ff', edgecolor='#ffffff')
plt.bar(X, -Y2, facecolor='#ff9999', edgecolor='#ffffff')

for x, y in zip(X, Y1):
    # ha: horizontal alignment
    plt.text(x + 0.1, y + 0.04, r'$%.2f$' % y, ha='center', va='bottom')
for x, y in zip(X, Y2):
    # ha: horizontal alignment
    plt.text(x + 0.1, -y - 0.04, r'$-%.2f$' % y, ha='center', va='top')
plt.xlim(-0.5, n)
plt.xticks(())
plt.ylim(-1.25, 1.25)
plt.yticks(())

plt.show()
