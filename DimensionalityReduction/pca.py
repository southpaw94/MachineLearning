# This script uses the 'primary component analysis' technique to 
# determine the k dimensions with the most variance of the 
# original set of d features.

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

from numpy.linalg import eig
import numpy as np

df_wine = pd.read_csv('wine.csv', header=None)
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Use the standard scaler to standardize the input data rather than
# remaking the wheel by writing a standardization function.
# Only the feature data needs to be standardized, recall that the output
# data is already discretized since we are primarily concerned with
# classification currently.
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)

# Create the covariance matrix from the transpose of the X_train_std data 
cov_mat = np.cov(X_train_std.T)

# Find the eigenvalues and eigenvectors of the covariance matrix
eig_vals, eig_vecs = eig(cov_mat)

total = sum(eig_vals)
var_exp = [(i / total) for i in sorted(eig_vals, reverse=True)]
var_exp_cumul = np.cumsum(var_exp)

plt.bar(range(1,14), var_exp, alpha=0.5, align='center', \
        label='Individual Explained Variance')
plt.step(range(1,14), var_exp_cumul, where='mid', \
        label='Cumulative Explained Variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal Components')
plt.legend(loc='best')
plt.show()

eig_pairs = [[np.abs(eig_vals[i]), eig_vecs[i]] for i in range(len(eig_vals))]
eig_pairs.sort(reverse=True)

w = np.hstack((eig_pairs[0][1][:, np.newaxis], eig_pairs[1][1][:, np.newaxis]))

print(X_train_std[0].dot(w))

# Transform training data into representative primary component analysis representation
X_train_pca = X_train_std.dot(w)

colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train == 1, 0], X_train_pca[y_train == 1, 1], c=c, label=l, marker=m)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.show()