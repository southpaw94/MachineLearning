# This program doesn't do much interesting,
# just creates a classification of the training
# data set using a binary decision tree

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import export_graphviz

from plots import PlotFigures

# Import and format all the data, pretty standard stuff
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# Create the binary decision tree using entropy
# as the impurity measurement (see Impurity.py),
# with a max search depth of 10
tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
tree.fit(X_train, y_train)

# Combine the test and train data
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

# Plot the combined data
PlotFigures.plot_decision_regions(X_combined_std, y_combined, classifier=tree, test_idx=range(105, 150))
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.show()

# Export this as tree.dot to the local directory
# This can then be converted to a png file with the command
# 'dot -Tpng tree.dot -o tree.png'
# For this command to execute, graphviz must be installed
# Available via the extra repository in Arch
export_graphviz(tree, out_file='tree.dot', \
        feature_names=['petal length', 'petal width'])
