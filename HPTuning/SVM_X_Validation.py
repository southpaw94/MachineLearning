import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

# This program introduces validation curves, which are essential
# in reducing over or under fitting of the learning algorithm.

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases'+\
        '/breast-cancer-wisconsin/wdbc.data', header=None)
X = df.loc[:, 2:].values
y = df.loc[:, 1].values

# All malignant tumors will be represented as class 1, otherwise, class 0
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, \
        test_size=0.20, random_state=1)

gs = GridSearchCV( \
        estimator = DecisionTreeClassifier(random_state = 0), \
        param_grid = [ \
            {'max_depth': [1, 2, 3, 4, 5, 6, 7, None]} \
        ], \
        scoring = 'accuracy', \
        cv = 5)

scores = cross_val_score(gs, \
        X_train, \
        y_train, \
        scoring = 'accuracy', \
        cv = 5)

print('CV accuracy: %.3f +/- %.3f' % ( \
        np.mean(scores), np.std(scores)))

