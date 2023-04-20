from scipy.sparse.linalg import eigsh
from sklearn.metrics import confusion_matrix, classification_report, normalized_mutual_info_score, adjusted_rand_score
import tensorflow as tf
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans


def similarity_matrix(data, sigma=1.0):
    sq_dists = squareform(pdist(data, 'sqeuclidean'))
    return np.exp(-sq_dists / (2 * sigma ** 2))


def laplacian_matrix(similarity_matrix):
    degree_matrix = np.diag(np.sum(similarity_matrix, axis=1))
    return degree_matrix - similarity_matrix


def one_step_adaptive_spectral_clustering2(X, num_clusters, k_nearest_neighbors, alpha=1.0, beta=1.0):
    n = X.shape[0]

    # Compute the pairwise Euclidean distances
    distances = squareform(pdist(X))

    # Find the k-nearest neighbors
    neighbors = np.argsort(distances)[:, 1:k_nearest_neighbors+1]

    # Compute the affinity matrix
    S = np.zeros((n, n))
    for i in range(n):
        for j in neighbors[i]:
            S[i, j] = np.exp(-distances[i, j] ** 2 / (2 * beta))
            S[j, i] = S[i, j]

    # Compute the degree matrix D
    D = np.diag(np.sum(S, axis=1))

    # Compute the normalized Laplacian matrix L
    D_inv_sqrt = np.diag(1 / np.sqrt(np.diagonal(D)))
    L = np.eye(n) - np.dot(D_inv_sqrt, np.dot(S, D_inv_sqrt))

    # Compute the first num_clusters eigenvectors of L
    _, H = eigsh(L, k=num_clusters, which='SM', maxiter=100000)

    # Normalize the rows of H
    H_normalized = H / np.linalg.norm(H, axis=1)[:, np.newaxis]

    # Apply K-means clustering to the rows of H_normalized
    kmeans = KMeans(n_clusters=num_clusters)
    labels = kmeans.fit_predict(H_normalized)

    return labels


def one_step_adaptive_spectral_clustering(data, num_clusters, k_neighbors=10):
    # Compute the similarity matrix
    sigma = 1.0
    S = similarity_matrix(data, sigma)

    # Compute the Laplacian matrix
    L = laplacian_matrix(S)

    # Perform eigendecomposition on the Laplacian matrix
    eigvals, eigvecs = np.linalg.eigh(L)

    # Select the k eigenvectors corresponding to the k smallest eigenvalues
    H = eigvecs[:, :num_clusters]
    H_norm = H / np.linalg.norm(H, axis=1)[:, np.newaxis]

    # Sort the indices of H_norm by their distance
    distances = squareform(pdist(H_norm, 'sqeuclidean'))
    sorted_indices = np.argsort(distances, axis=1)

    # Update the affinity matrix
    S_hat = np.zeros_like(S)
    for i in range(S.shape[0]):
        # Exclude the first index, which corresponds to the point itself
        idx = sorted_indices[i, 1:k_neighbors + 1]
        d_sum = np.sum(distances[i, idx])
        kth_distance = distances[i, idx[-1]]
        S_hat[i, idx] = (kth_distance - distances[i, idx]) / \
            (k_neighbors * kth_distance - d_sum)

    # Compute the updated Laplacian matrix
    L_hat = laplacian_matrix(S_hat)

    # Perform eigendecomposition on the updated Laplacian matrix
    eigvals_hat, eigvecs_hat = np.linalg.eigh(L_hat)

    # Select the k eigenvectors corresponding to the k smallest eigenvalues
    H_hat = eigvecs_hat[:, :num_clusters]
    epsilon = 1e-8
    H_hat_norm = H_hat / (np.linalg.norm(H_hat, axis=1)
                          [:, np.newaxis] + epsilon)
    # Cluster the data using k-means
    kmeans = KMeans(n_clusters=num_clusters).fit(H_hat_norm)

    # Return the cluster labels
    return kmeans.labels_


# Load the MNIST dataset
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()

# Preprocess the dataset
x_train = x_train.reshape(-1, 28 * 28) / 255.0

# Take a subset of the dataset
num_samples = 6000
x_train_subset = x_train[:num_samples]
y_train_subset = y_train[:num_samples]
# Perform One-Step Adaptive Spectral Clustering
num_clusters = 10
k_neighbors = 10
labels = one_step_adaptive_spectral_clustering(
    x_train_subset, num_clusters, k_neighbors)

# Evaluate the performance
conf_matrix = confusion_matrix(y_train_subset, labels)
report = classification_report(y_train_subset, labels)

print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(report)

nmi = normalized_mutual_info_score(y_train_subset, labels)
ari = adjusted_rand_score(y_train_subset, labels)

print("Normalized Mutual Information (NMI):", nmi)
print("Adjusted Rand Index (ARI):", ari)
