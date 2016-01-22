from pandas import DataFrame
from scipy.io import loadmat

import matplotlib.pyplot as plt
import numpy as np

data = loadmat('euclideanD.mat')['euclideanD']
data = np.reshape(data, (200, 2))

df = DataFrame(data, columns=['a', 'b'])
df.plot(kind='hexbin', x='a', y='b')
plt.show()
