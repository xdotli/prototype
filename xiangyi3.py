import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import eig
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits


def calculate_distance_matrix(X):
    return cdist(X, X, metric='euclidean')


def construct_laplacian_from_distance_matrix(distance_matrix):
    return distance_matrix


def one_step_adaptive_spectral_clustering(data, num_clusters, k_neighbors):
    distance_matrix = calculate_distance_matrix(data)
    Laplacian = construct_laplacian_from_distance_matrix(distance_matrix)

    # Compute the eigenvectors corresponding to the smallest num_clusters eigenvalues
    eigenvalues, eigenvectors = eig(Laplacian)
    idx = np.argsort(eigenvalues)[:num_clusters]
    H = np.real(eigenvectors[:, idx])

    # Normalize the rows of H
    H_norm = H / np.linalg.norm(H, axis=1)[:, np.newaxis]

    # Perform k-means clustering on the normalized matrix H
    kmeans = KMeans(n_clusters=num_clusters).fit(H_norm)
    labels = kmeans.labels_

    return labels


# Load the MNIST dataset
digits = load_digits()
data = digits.data
targets = digits.target

# Use a subset of the MNIST dataset
subset_size = 1000
indices = np.random.choice(len(data), subset_size, replace=False)
data_subset = data[indices]
targets_subset = targets[indices]

# Apply the modified one_step_adaptive_spectral_clustering algorithm
num_clusters = 10
k_neighbors = 10
labels = one_step_adaptive_spectral_clustering(
    data_subset, num_clusters, k_neighbors)

# Calculate the accuracy
accuracy = np.sum(labels == targets_subset) / subset_size
print("Accuracy:", accuracy)
