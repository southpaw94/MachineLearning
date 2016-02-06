import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from sklearn.learning_curve import learning_curve, validation_curve
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC

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

pipe_svc = Pipeline([('scl', StandardScaler()), \
        ('clf', SVC(random_state=1))])

param_range = [10**i for i in range(-4, 4)]
param_grid = [{'clf__C': param_range, \
        'clf__kernel': ['linear']}, \
        {'clf__C': param_range, \
        'clf__gamma': param_range, \
        'clf__kernel': ['rbf']}]

gs = GridSearchCV(estimator=pipe_svc, \
        param_grid=param_grid, \
        scoring='accuracy', \
        cv=10, \
        n_jobs=-1)

gs = gs.fit(X_train, y_train)
print('Best score from exhaustive search: %.3f' % gs.best_score_)
print('Best parameter set from exhaustive search: %s' % gs.best_params_)

# How does this best model fit our test data?
clf = gs.best_estimator_
clf.fit(X_train, y_train)
print('Grid search test accuracy: %.3f' % clf.score(X_test, y_test))

# How does nested cross-validation perform?
scores = cross_val_score(gs, X, y, scoring='accuracy', cv = 10)
print('CV accuracy: %.3f +/- %.3f' %(np.mean(scores), \
        np.std(scores)))
