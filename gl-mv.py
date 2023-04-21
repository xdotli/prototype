import numpy as np
import graphlearning as gl
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load MNIST labels and results of k-nearest neighbor search
labels = gl.datasets.load('mnist', labels_only=True)
W = gl.weightmatrix.knn('mnist', 10, metric='vae')

# Run spectral clustering
model = gl.clustering.spectral(W, num_clusters=10, extra_dim=4)
cluster_labels = model.fit()

# Check accuracy
acc = gl.clustering.clustering_accuracy(cluster_labels, labels)
print('Accuracy = %.2f%%' % acc)

# Calculate the Normalized Mutual Information (NMI) as a performance metric
nmi = normalized_mutual_info_score(labels, cluster_labels)
print("Normalized Mutual Information:", nmi)

# Calculate the Adjusted Rand Index (ARI) as a performance metric
ari = adjusted_rand_score(labels, cluster_labels)
print("Adjusted Rand Index:", ari)

# Visualize the clustering results
X = gl.datasets.load('mnist')
X_embedded = TSNE(n_components=2).fit_transform(X)

plt.figure(figsize=(10, 8))
cmap = plt.cm.get_cmap('tab10', 10)
scatter = plt.scatter(
    X_embedded[:, 0], X_embedded[:, 1], c=cluster_labels, cmap=cmap, alpha=0.5)
plt.title("Spectral Clustering Visualization")
plt.colorbar(scatter, ticks=range(10), label='Cluster Label')
plt.show()
