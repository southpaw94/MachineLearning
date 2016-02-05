from pca_kernel import rbf_kernel_pca
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from sklearn.datasets import make_moons
from sklearn.decomposition import PCA
import numpy as np

def project_x(x_new, X, gamma, alphas, lambdas):
    pair_dist = np.array([np.sum((x_new - row)**2) for row in X])
    k = np.exp(-gamma * pair_dist)
    return k.dot(alphas / lambdas)

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
X_kpca, eigvals = rbf_kernel_pca(X, gamma=15, n_components=2)
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

# Grab a random point, put it through the PCA, and be able to get the original point 
# back from just the projection matrix
alphas, lambdas = rbf_kernel_pca(X, gamma=15, n_components = 1)
# Original point
x_new = X[25]
# projected point
x_proj = alphas[25]

# Original point from projected point and projection parameters
x_reproj = project_x(x_new, X, gamma=15, alphas=alphas, lambdas=lambdas)

# Plot these three points to show they are the same
plt.scatter(alphas[y==0, 0], np.zeros((50)), color='red', marker='^', alpha=0.5)
plt.scatter(alphas[y==1, 0], np.zeros((50)), color='blue', marker='o', alpha=0.5)
plt.scatter(x_proj, 0, color='black', label='original projection of point X[25]', marker='^', s=100)
plt.scatter(x_reproj, 0, color='green', label = 'remapped point X[25]', marker='x', s=500)
plt.legend(scatterpoints=1)
plt.show()
