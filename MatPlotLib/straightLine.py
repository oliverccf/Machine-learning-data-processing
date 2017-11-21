import matplotlib.pyplot as plt
import numpy as np

x1 = np.array([1,2,3])
y1 = x1

x2 = x1+1
y2 = y1+1

plt.plot(x1, y1, color='green', linestyle='dashed', marker='o',
     markerfacecolor='blue', markersize=12)

plt.show()