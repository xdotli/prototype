import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.datasets import fetch_olivetti_faces
from sklearn.metrics import normalized_mutual_info_score

# Load the Yale Face dataset
faces = fetch_olivetti_faces()
X = faces.data / 255.0
y = faces.target

# Apply spectral clustering with 20 clusters
n_clusters = 20
sc = SpectralClustering(n_clusters=n_clusters, affinity='rbf', gamma=1.0)
y_pred = sc.fit_predict(X)

# Compute NMI score between true labels and predicted labels
nmi_score = normalized_mutual_info_score(y, y_pred)

print("NMI score: {:.2f}".format(nmi_score))

# Plot some example faces and their predicted labels
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i in range(5):
    ax = axes[0, i]
    ax.imshow(X[i].reshape(64, 64), cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("True label: {}".format(y[i]))
    
    ax = axes[1, i]
    ax.imshow(X[i].reshape(64, 64), cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Pred label: {}".format(y_pred[i]))

plt.show()

