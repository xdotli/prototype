import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.neighbors import kneighbors_graph

# Load the MNIST dataset from TensorFlow
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape((x_train.shape[0], -1))
x_train = x_train.astype(np.float32) / 255.0

# Define the parameters for the One-Step Adaptive Spectral Clustering algorithm
n_clusters = 10
n_neighbors = 10
sigma = 1.0
alpha = 1.0
beta = 1.0

# Construct the affinity matrix S using the KNN algorithm
knn_graph = kneighbors_graph(
    x_train, n_neighbors=n_neighbors, mode='connectivity')
knn_graph = 0.5 * (knn_graph + knn_graph.T)
S = np.exp(-knn_graph.power(2).toarray() / (2.0 * sigma ** 2))

# Compute the Laplacian matrix L
D = np.diag(np.sum(S, axis=1))
L = D - S

# Compute the eigenvectors and eigenvalues of L
eigenvalues, eigenvectors = np.linalg.eigh(L)

# Choose the first n_clusters eigenvectors corresponding to the smallest eigenvalues
H = eigenvectors[:, :n_clusters]

# Apply K-means clustering to the rows of H to get the initial cluster assignments
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(H)
Y = kmeans.labels_

# Apply the One-Step Adaptive Spectral Clustering algorithm
for i in range(10):
    # Compute the pairwise distances between the rows of H
    distances = np.sum(
        (H[:, np.newaxis, :] - H[np.newaxis, :, :]) ** 2, axis=-1)
    # Compute the diagonal and off-diagonal terms of the affinity matrix
    diagonal = np.diag(distances)
    off_diagonal = distances - diagonal
    # Compute the adaptive affinity matrix
    S_hat = np.zeros_like(S)
    for j in range(n_clusters):
        indices = np.where(Y == j)[0]
        if len(indices) > 0:
            kth_distance = np.partition(off_diagonal[indices][:, indices], kth=n_neighbors)[
                :, n_neighbors - 1]
            numerator = kth_distance[:, np.newaxis] - \
                off_diagonal[indices][:, indices]
            denominator = np.sum(kth_distance) - \
                np.sum(off_diagonal[indices][:, indices], axis=1)
            denominator[denominator == 0] = np.finfo(float).eps
            S_hat[np.ix_(indices, indices)] = numerator / \
                denominator[:, np.newaxis]
    S_hat = 0.5 * (S_hat + S_hat.T)
    # Compute the optimal rotation matrix R
    U, _, Vt = np.linalg.svd(Y[:, np.newaxis] == np.arange(
        n_clusters)[np.newaxis, :], full_matrices=False)
    R = np.dot(Vt.T, U.T)
    # Update the cluster assignments Y
    Y = np.argmax(np.dot(H, R), axis=1)
    # Update the spectral embedding H
    H = np.dot(np.linalg.inv(np.dot(R.T, R)), np.dot(R.T, np.dot(
        H, S_hat + alpha * np.dot(Y[:, np.newaxis], Y[np.newaxis, :]) + beta * S)))


nmi = normalized_mutual_info_score(y_train, Y)
print(f"Normalized Mutual Information: {nmi:.4f}")


fig, axs = plt.subplots(nrows=n_clusters, figsize=(8, 16))
for i in range(n_clusters):
    indices = np.where(Y == i)[0]
    axs[i].set_title(f"Cluster {i}")
    axs[i].imshow(np.mean(x_train[indices], axis=0).reshape(
        28, 28), cmap='gray')
    axs[i].axis('off')
plt.tight_layout()
plt.show()
