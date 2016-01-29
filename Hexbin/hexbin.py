from pandas import DataFrame
from scipy.io import loadmat
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np

x = np.arange(400) + 1
y = loadmat('euclideanD.mat')['euclideanD']
y = np.reshape(y, (400, 1))
z = np.zeros((400, 2))
z = np.column_stack((x, y))

df = DataFrame(z, columns=['a', 'b'])
df.plot(kind='hexbin', x='a', y='b', gridsize=25)
plt.show()
