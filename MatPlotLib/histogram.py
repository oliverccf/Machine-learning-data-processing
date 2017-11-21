import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

np.random.seed(19680801)

# example data
mu = 100  # mean of distribution
sigma = 15  # standard deviation of distribution
x = mu + sigma * np.random.randn(437)

plt.hist(x, bins=200)

plt.show()