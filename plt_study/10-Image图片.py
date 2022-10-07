import numpy as np
import matplotlib.pyplot as plt

a = np.linspace(0, 1, 81).reshape(9, 9)
print(a)
plt.imshow(a,
           interpolation='nearest',
           cmap='bone',
           origin='upper',
           )
plt.colorbar()  # 颜色对比框

plt.show()