import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score

# This program does the same thing as SKPipeline.py, but uses
# the built in sklearn function cross_val_score which removes
# the necessity for the custom iterations through each fold.

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases'+\
        '/breast-cancer-wisconsin/wdbc.data', header=None)
X = df.loc[:, 2:].values
y = df.loc[:, 1].values

# All malignant tumors will be represented as class 1, otherwise, class 0
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, \
        test_size=0.20, random_state=1)

pipe_lr = Pipeline([('scl', StandardScaler()), ('pca', PCA(n_components=2)), \
        ('clf', LogisticRegression(random_state=1))])
pipe_lr.fit(X_train, y_train)

print('Test Accuracy: %.3f' % pipe_lr.score(X_test, y_test))

# Note that the n_jobs parameter in the cross_val_score call is used
# to parallelize the task of the computation, if you are on a multi-core
# system, increase this value to reflect that. Note that for large
# training sets, if the n_jobs value is set very high, this could bring
# your system to a grinding halt.
scores = cross_val_score(estimator = pipe_lr, X = X_train, \
        y = y_train, cv = 10, n_jobs = 1)

print('CV accuracy scores: %s' % scores)

print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
