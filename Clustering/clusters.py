import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

"""
This program fits the k-means clustering algorithm to the blobs
dataset with varying values of n_clusters parameter, to plot the
distortion of the clusters versus the number of clusters
selected. The point at which greatest change in slope occurs is
the elbow point, representative of the proper choice for 
n_clusters.
"""

def main():

    X, y = make_blobs(n_samples=150,
            n_features=2,
            centers=3,
            cluster_std=0.5,
            shuffle=True,
            random_state=0)

    distortions = []

    for i in range(1, 11):
        km = KMeans(n_clusters=i,
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=0)
        km.fit(X)
        distortions.append(km.inertia_)

    plt.plot(range(1, 11), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.annotate('Elbow point \n(number of clusters)', (3, distortions[2]))
    plt.show()
    return

if __name__ == '__main__':
    main()
