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

plt.clf()
scikit_pca = PCA(n_components=2)
X_spca = scikit_pca.fit_transform(X)
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
ax[0].set_title('Attempt at linear classification region')
plt.show()
