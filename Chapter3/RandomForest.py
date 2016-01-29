from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from matplotlib.colors import ListedColormap

import numpy as np
import matplotlib.pyplot as plt

from plots import PlotFigures

# Random forest classifiers make use of the idea behind
# binary tree classification, but they do so in a parellel
# way by creating a certain amount of binary trees with
# a different set of randomly selected training data
# each time.

iris = datasets.load_iris()
X = iris.data[:, [2,3]]
y = iris.target

X_train, X_test, y_train, y_test = \
         train_test_split(X, y, test_size=0.3, random_state=0)

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

# n_estimators defaults to 10, so we are training 10
# decision trees via 2 processor cores (n_jobs = 2)
forest = RandomForestClassifier(criterion='entropy', random_state = 1, n_jobs=2)
forest.fit(X_train, y_train)

PlotFigures.plot_decision_regions(X_combined, y_combined, classifier=forest, test_idx=range(105, 150))
plt.show()

for X_test_val in X_test:
    print(forest.predict_proba(X_test_val.reshape(1, -1)))
