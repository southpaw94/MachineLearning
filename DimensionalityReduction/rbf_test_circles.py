from sklearn.datasets import make_circles
from sklearn.decomposition import PCA
from pca_kernel import rbf_kernel_pca
import matplotlib.pyplot as plt
import numpy as np

X, y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)
plt.scatter(X[y==0, 0], X[y==0, 1], color='red', marker='^', alpha=0.5)
plt.scatter(X[y==1, 0], X[y==1, 1], color='blue', marker='o', alpha=0.5)
plt.show()

# Make an attempt at a linear PCA analysis
scikit_pca = PCA(n_components = 2)
X_spca = scikit_pca.fit_transform(X)
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (7, 3))

# Create scatter plots
plt.suptitle('Linear PCA')

ax[0].scatter(X_spca[y==0, 0], X_spca[y == 0, 1], color='red', marker='^', alpha=0.5)
ax[0].scatter(X_spca[y==1, 0], X_spca[y == 1, 1], color='blue', marker='o', alpha=0.5)

ax[1].scatter(X_spca[y==0, 0], np.zeros((500, 1)) + 0.02, color='red', marker='^', alpha=0.5)
ax[1].scatter(X_spca[y==1, 0], np.zeros((500, 1)) - 0.02, color='blue', marker='o', alpha=0.5)

# Set the first subplot properties
ax[0].set_title('PC2 vs PC1')
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')

# Set the second subplot properties
ax[1].set_title('Flattened to PC1')
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
ax[1].set_ylim([-1,1])

plt.show()

# Once again, the linear PCA analysis clearly does not work
# let's try the radial basis function

# Apply the radial basis function
X_kpca = rbf_kernel_pca(X, gamma=15, n_components=2)

# Create the scatter plots
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7,3))
plt.suptitle('RBF Kernel PCA')

ax[0].scatter(X_kpca[y==0, 0], X_kpca[y==0, 1], color='red', marker='^', alpha=0.5)
ax[0].scatter(X_kpca[y==1, 0], X_kpca[y==1, 1], color='blue', marker='o', alpha=0.5)
ax[1].scatter(X_kpca[y==0, 0], np.zeros((500,1)) + 0.02, color='red', marker='^', alpha=0.5)
ax[1].scatter(X_kpca[y==1, 0], np.zeros((500,1)) - 0.02, color='blue', marker='o', alpha=0.5)

# Format the first subplot
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[0].set_title('PC2 vs PC1')

# Format the second subplot
ax[1].set_xlabel('PC1')
ax[1].set_yticks([])
ax[1].set_ylim([-1, 1])
ax[1].set_title('Flattened to PC1')

plt.show()


