from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from plots import PlotFigures
from matplotlib.colors import ListedColormap

import numpy as np
import matplotlib.pyplot as plt

# This project provides the same capabilities as the LogisticRegression.py program
# but uses a radial basis function kernel for the learning algorithm rather than
# a sigma function. This allows it to group data into sets with non-linear
# boundaries.

iris = datasets.load_iris()
X = iris.data[:, [2,3]]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

g = float(input("Input gamma: "))

svm = SVC(C=1.0, kernel='rbf',gamma=g, random_state = 0, probability=True)
svm.fit(X_train_std, y_train)

PlotFigures.plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105, 150))
plt.show()

for X_test_val in X_test_std:
    print(svm.predict_proba(X_test_val.reshape(1, -1)))
