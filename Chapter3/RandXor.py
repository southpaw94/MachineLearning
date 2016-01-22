from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from matplotlib.colors import ListedColormap

import numpy as np
import matplotlib.pyplot as plt

from plots import PlotFigures

# This program is a simulation of a data set with a non-linear separation region.
# The data is gathered via numpy's randn command, which is not truly random, so the data
# will be the same through each simulation. Data is segregated by quadrant, X,Y can take
# four different quadrants. As the data is collected via the randn function, which returns
# values with mean 0 and standard deviation 1, all points lie inside the range -3, 3, which
# is purely coincidental since this is not a true random function and is seeded with the
# same number each time (0). Scikit-learn then makes use of the SVC function, which
# organizes this data into two distinct groups. Finally, matplotlib.pyplot plots the 
# colored output data.

np.random.seed(0)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)
g = float(input("Input gamma: "))
# The gamma value here is effectively a cutoff parameter for a normal distribution sphere
# which is used to vary the decision boundary between groups.
# See the help page for more details: help(sklearn.svm.SVC)
svm = SVC(kernel='rbf', C=10.0, gamma=g, random_state = 0)
svm.fit(X_xor, y_xor)

PlotFigures.plot_decision_regions(X_xor, y_xor, classifier=svm, test_idx=range(105, 150))
plt.show()

#for X_test_val in X_test_std:
#    print(lr.predict_proba(X_test_val.reshape(1, -1)))
