# A test of the SBS algorithm in sbs.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sbs import SBS as sbs
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Import the wines data set
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)

# Assign column names to the pandas DataFrame
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', \
        'Ash', 'Alcalinity of ash', 'Magnesium', \
        'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', \
        'Proanthocyanins', 'Color intensity', 'Hue', \
        'OD280/OD315 of diluted wines', 'Proline']

# Split the data set into features and outputs
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:,0].values

# Split the feature and output datasets into training and
# testing components, 30% for testing, rest for training
X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.3, random_state=0)

# Standardize data (x_std = (x - x_avg) / x_dev
ss = StandardScaler()
X_train_std = ss.fit_transform(X_train)
X_test_std = ss.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=2)
sbs = sbs(knn, k_features=1)
sbs.fit(X_train_std, y_train)

k_feat = [len(k) for k in sbs.subsets_]
plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.1])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.show()

# For comparison purposes: original data fit vs feature
# selection data fit

# By examining the output, we can tell that test subset 8
# has a very good fit
k5 = list(sbs.subsets_[8])

print('Original training with full dataset')
knn.fit(X_train_std, y_train)
print('Training accuracy (training data):', knn.score(X_train_std, y_train))
print('Training accuracy (test data):', knn.score(X_test_std, y_test))

print('Sequential backward selection training')
knn.fit(X_train_std[:, k5], y_train)
print('Training accuracy (training data):', knn.score(X_train_std[:, k5], y_train))
print('Training accuracy (test data):', knn.score(X_test_std[:, k5], y_test))
