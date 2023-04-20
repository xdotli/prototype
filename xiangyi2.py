from scipy.optimize import linear_sum_assignment
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from scipy.sparse.linalg import lobpcg
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras.datasets import mnist


def similarity_matrix(data, sigma=1.0):
    sq_dists = squareform(pdist(data, 'sqeuclidean'))
    return np.exp(-sq_dists / (2 * sigma ** 2))


def laplacian_matrix(similarity_matrix):
    degree_matrix = np.diag(np.sum(similarity_matrix, axis=1))
    return degree_matrix - similarity_matrix


def spectral_clustering(data, num_clusters, sigma=1.0):
    S = similarity_matrix(data, sigma)
    L = laplacian_matrix(S)
    L = csr_matrix(L)

    # Random initial guess for eigenvectors
    np.random.seed(0)
    X = np.random.rand(L.shape[0], num_clusters)

    # Compute the eigenvalues and eigenvectors of L
    eigenvalues, eigenvectors = lobpcg(L, X, largest=False)

    # Select the k eigenvectors corresponding to the k smallest eigenvalues
    X = eigenvectors

    # Normalize rows of X (optional)
    X_norm = X / np.linalg.norm(X, axis=1)[:, np.newaxis]

    # Cluster the data using k-means
    kmeans = KMeans(n_clusters=num_clusters).fit(X_norm)

    # Return the cluster labels
    return kmeans.labels_


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
