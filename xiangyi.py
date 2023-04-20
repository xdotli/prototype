from scipy.optimize import linear_sum_assignment
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from scipy.sparse.linalg import eigsh

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import StandardScaler


def similarity_matrix(data, sigma=1.0):
    sq_dists = squareform(pdist(data, 'sqeuclidean'))
    return np.exp(-sq_dists / (2 * sigma ** 2))


def laplacian_matrix(similarity_matrix):
    degree_matrix = np.diag(np.sum(similarity_matrix, axis=1))
    return degree_matrix - similarity_matrix


def spectral_clustering(data, num_clusters, sigma=1.0):
    # Step 1: Construct the similarity matrix
    S = similarity_matrix(data, sigma)

    # Step 2: Compute the Laplacian matrix
    L = laplacian_matrix(S)

    # Step 3: Compute the eigenvalues and eigenvectors of L
    eigenvalues, eigenvectors = eigsh(L, k=num_clusters, which='SM')

    # Step 4: Select the k eigenvectors corresponding to the k smallest eigenvalues
    X = eigenvectors

    # Step 5: Normalize rows of X (optional)
    X_norm = X / np.linalg.norm(X, axis=1)[:, np.newaxis]

    # Step 6: Cluster the data using k-means
    kmeans = KMeans(n_clusters=num_clusters).fit(X_norm)

    # Step 7: Return the cluster labels
    return kmeans.labels_


# Example usage
data = np.array([[1, 1], [1, 2], [2, 1], [2, 2],
                [8, 8], [8, 9], [9, 8], [9, 9]])
num_clusters = 2

labels = spectral_clustering(data, num_clusters)
print(labels)


# Load and preprocess the MNIST dataset
(x_train, y_train), (_, _) = mnist.load_data()

# Flatten the images and standardize the data
x_train = x_train.reshape(-1, 28*28)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train.astype(np.float64))

# Select a small subset of the data
sample_size = 500
indices = np.random.choice(len(x_train), sample_size, replace=False)
x_train_subset = x_train[indices]
y_train_subset = y_train[indices]

# Apply spectral clustering to the subset of the MNIST dataset
num_clusters = 10
labels = spectral_clustering(x_train_subset, num_clusters)

# Evaluate the clustering results
conf_matrix = confusion_matrix(y_train_subset, labels)
print("Confusion Matrix:")
print(conf_matrix)

# Calculate the accuracy score (using Hungarian Algorithm for optimal matching)

row_ind, col_ind = linear_sum_assignment(-conf_matrix)
accuracy = conf_matrix[row_ind, col_ind].sum() / sample_size
print("Accuracy:", accuracy)
