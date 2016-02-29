import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

"""
This program makes a set of three clusters with two features.
The KMeans method of sklearn.clusters is then used to classify
points into one of three clusters, with no knowledge of which
these actually belong to (unsupervised learning). 

Note: An inherent weakness of k-means clustering is that for
an accurate model, the number of clusters must be given to
the algorithm. To find the ideal value for this (in an unknown
situation), the methodology in clusters.py can be used.
"""

# Font dictionary for matplotlib
font = {'family': 'normal',
        'weight': 'normal',
        'size': 22}

def main():

    # Create our blobs, the y array is not used as this is a
    # demonstration of unsupervised learning
    X, y = make_blobs(n_samples=150,
            n_features=2,
            centers=3,
            cluster_std=0.5,
            shuffle=True,
            random_state=0)

    # Create our k-means classifier
    km = KMeans(n_clusters=3,
            init='k-means++',
            n_init=10,
            max_iter=300,
            tol=1e-04,
            random_state=0)

    # Fit the X data set with the k-means clustering algorithm
    y_km = km.fit_predict(X)

    # Plot everything
    plt.scatter(X[y_km == 0, 0], X[y_km == 0, 1],
            marker='o',
            color='r',
            label='Cluster 1',
            s=100)

    plt.scatter(X[y_km == 1, 0], X[y_km == 1, 1],
            marker='v',
            color='g',
            label='Cluster 2',
            s=100)

    
    plt.scatter(X[y_km == 2, 0], X[y_km == 2, 1],
            marker='s',
            color='orange',
            label='Cluster 3',
            s=100)

    # This section plots the centroids of the final clusters
    plt.scatter(km.cluster_centers_[:, 0],
            km.cluster_centers_[:, 1],
            marker='*',
            c='yellow',
            s=500,
            label='Centroids')

    plt.rc('font', **font)
    plt.legend(loc='lower left')
    plt.show()
    return

if __name__ == '__main__':
    main()
