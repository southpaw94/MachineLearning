import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform

variables = ['X', 'Y', 'Z']
labels = ['ID_0', 'ID_1', 'ID_2', 'ID_3', 'ID_4']

np.random.seed(123)
X = np.random.random_sample([5, 3]) * 10
df = pd.DataFrame(X, columns=variables, index=labels)
dist = pd.DataFrame(squareform(pdist(X, metric='euclidean')), columns=labels, index=labels)

row_clusters = linkage(df.values,
        method='complete',
        metric='euclidean')

row_clusters_df = pd.DataFrame(row_clusters,
        columns=['row label 1', 'row label 2', 'distance', 'no. of items in cluster'],
        index=['cluster %d' % i for i in range(row_clusters.shape[0])])

fig = plt.figure()
axd = fig.add_axes([0.09, 0.1, 0.2, 0.6])
row_dendr = dendrogram(row_clusters, orientation='left')

df_rowclust = df.ix[row_dendr['leaves'][::-1]]

axm = fig.add_axes([0.05, 0.1, 0.6, 0.6])
cax = axm.matshow(df_rowclust,
        interpolation='nearest', cmap='hot_r')

axd.set_xticks([])
axd.set_yticks([])
for i in axd.spines.values():
    i.set_visible(False)
fig.colorbar(cax)
axm.set_xticklabels([''] + list(df_rowclust.columns))
axm.set_yticklabels([''] + list(df_rowclust.index))
plt.show()
