from pca_kernel import rbf_kernel_pca
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from sklearn.datasets import make_moons
from sklearn.decomposition import PCA
import numpy as np

# Create and plot the raw half moon data
X, y = make_moons(n_samples = 100, random_state = 123)
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', \
        marker='^', alpha=0.5)
plt.scatter(X[y==1, 0], X[y==1, 1], color='blue', \
        marker='o', alpha=0.5)
plt.title('Raw Data')
plt.show()

# Create a linear principle component analysis class with 2 principle components
scikit_pca = PCA(n_components=2)
X_spca = scikit_pca.fit_transform(X)

# Create a 1x2 plot showing the raw linearly transformed data
# and the flattened linearly transformed data
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))
ax[0].scatter(X_spca[y==0, 0], X_spca[y==0, 1], color='red', \
        marker='^', alpha=0.5)
ax[0].scatter(X_spca[y==1, 0], X_spca[y==1, 1], color='blue', \
        marker='o', alpha=0.5)
ax[1].scatter(X_spca[y==1, 0], np.zeros((50,1)) + 0.02, color='red', \
        marker='^', alpha=0.5)
ax[1].scatter(X_spca[y==1, 0], np.zeros((50, 1)) - 0.02, color='blue', \
        marker='o', alpha=0.5)
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC1')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
plt.suptitle('Attempt at linear classification region')
plt.show()

# Using our radial basis function kernel, create a non-linear transformation matrix
# This will transform the data into some other dimension where it is linearly 
# seperable, and then select the k primary components from this result
X_kpca = rbf_kernel_pca(X, gamma=15, n_components=2)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7,3))
ax[0].scatter(X_kpca[y==0, 0], X_kpca[y==0, 1], color='red', marker='^', alpha=0.5)
ax[0].scatter(X_kpca[y==1, 0], X_kpca[y==1, 1], color='blue', marker='o', alpha=0.5)
ax[1].scatter(X_kpca[y==0, 0], np.zeros((50, 1)) + 0.02, color='red', marker='^', alpha=0.5)
ax[1].scatter(X_kpca[y==1, 0], np.zeros((50, 1)) - 0.02, color='blue', marker='o', alpha=0.5)
ax[0].set_xlabel('PC1')
ax[1].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_yticks([])
ax[1].set_ylim([-1, 1])
ax[0].set_title('RBF Principle Component Analysis')
ax[1].set_title('Flattened to only show PC1')
plt.show()
