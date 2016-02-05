from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

# This program uses sklearn's inherent kernel PCA algorithm with the 
# radial basis function to find linear mappings for non-linear separation
# regions. It then displays four plots, at various values for gamma, which
# displays one of the inherent weaknesses of kernel functions, that
# the proper value for gamma requires some trial and error to find.

# Use scikit to find the kernel PCA for 2 eigenvectors
# using the same half moon data as previously
X, y = make_moons(n_samples=100, random_state=123)
scikit_kpca = [KernelPCA(n_components=2, kernel='rbf', gamma=n*5+5) for n in range(4)] 
X_skernpca = [scikit_kpca[n].fit_transform(X) for n in range(4)]

fig, ax = plt.subplots(nrows=2, ncols=2)

# Plot this representation to see if it concurs with our custom
# kernel PCA algorithm
for i in range(4):
    temp = X_skernpca[i]
    ax[0 if i < 2 else 1, i % 2].scatter(temp[y==0, 0], temp[y==0, 1], color='red', marker='^', alpha=0.5)
    ax[0 if i < 2 else 1, i % 2].scatter(temp[y==1, 0], temp[y==1, 1], color='blue', marker='o', alpha=0.5)
    ax[0 if i < 2 else 1, i % 2].set_xlabel('PC1')
    ax[0 if i < 2 else 1, i % 2].set_ylabel('PC2')
    ax[0 if i < 2 else 1, i % 2].set_title('Gamma = ' + str(i * 5 + 5))
    
plt.suptitle('RBF kernel PCA for various gamma values')
plt.show()
