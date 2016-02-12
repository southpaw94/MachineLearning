import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split

raw_data = datasets.load_breast_cancer()
sc = StandardScaler()
pca = PCA(n_components = None)

X, y, features = raw_data.data, \
        raw_data.target, \
        raw_data.feature_names

X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.2)

X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

variance = pca.explained_variance_ratio_
cumul_variance = np.cumsum(variance)

plt.bar(range(1, len(features) + 1), variance, label="Explained Variance Ratio", align='center')
plt.step(range(1, len(features) + 1), cumul_variance, where='post', label="Cumulative Variance")
plt.xticks(range(1, len(features) + 1), features, rotation='vertical')
plt.subplots_adjust(bottom=0.35)
plt.show()

for variance, label in zip(cumul_variance, features):
    if variance < 0.9:
        print('%s is a principal component with weight: %0.2f' % (label, variance))
