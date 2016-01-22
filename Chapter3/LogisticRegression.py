from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from matplotlib.colors import ListedColormap

import numpy as np
import matplotlib.pyplot as plt

from plots import PlotFigures

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

lr = LogisticRegression(C=1000.0, random_state = 0)
lr.fit(X_train_std, y_train)

PlotFigures.plot_decision_regions(X_combined_std, y_combined, classifier=lr, test_idx=range(105, 150))
plt.show()

for X_test_val in X_test_std:
    print(lr.predict_proba(X_test_val.reshape(1, -1)))
