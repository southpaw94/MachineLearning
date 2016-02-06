import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

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

pipe_svc.fit(X_train, y_train)
y_pred = pipe_svc.predict(X_test)
confmat = confusion_matrix(y_true = y_test, y_pred = y_pred)
print(confmat)

fig, ax = plt.subplots(figsize = (2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x = j, y = i, \
                s = confmat[i, j], 
                va = 'center', ha = 'center')

plt.xlabel('predicted label')
plt.ylabel('true label')
plt.show()
