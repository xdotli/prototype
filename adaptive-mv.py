import matplotlib.pyplot as plt
from scipy.stats import mode
from scipy.spatial.distance import squareform, pdist
from sklearn.metrics import accuracy_score, normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import tensorflow as tf
from tensorflow.keras.datasets import mnist


def compute_affinity_matrix(X, sigma):
    pairwise_distances = squareform(pdist(X, metric='euclidean'))
    A = np.exp(-pairwise_distances ** 2 / (2 * sigma ** 2))
    np.fill_diagonal(A, 0)
    return A


def adaptive_sparse_graph_learning(Xs, n_clusters, alpha, sigma):
    # Compute affinity matrices
    Ws = [compute_affinity_matrix(X, sigma) for X in Xs]

    # Compute degree matrices
    Ds = [np.diag(np.sum(W, axis=1)) for W in Ws]

    # Compute Laplacian matrices
    Ls = [D - W for D, W in zip(Ds, Ws)]

    # Compute eigenvectors of Laplacian matrices
    eig_vals_list, eig_vecs_list = [], []
    for L in Ls:
        eig_vals, eig_vecs = np.linalg.eig(L)
        eig_vals_list.append(eig_vals)
        eig_vecs_list.append(eig_vecs)

    # Determine the shared eigenvectors
    n_nodes = Xs[0].shape[0]
    Hs = [normalize(np.real(eig_vecs[:, np.argsort(eig_vals)[1:(n_clusters + 1)]]))
          for eig_vals, eig_vecs in zip(eig_vals_list, eig_vecs_list)]
    H_new = np.zeros((n_nodes, n_clusters))

    for i in range(n_nodes):
        H_new[i, :] = (alpha * Hs[0][i, :] + (1 - alpha)
                       * Hs[1][i, :]).sum(axis=0)

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(H_new)

    return kmeans.labels_, H_new


def create_views(X, n_component):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    tsne = TSNE(n_components=n_components, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X)

    return [X_pca, X_tsne]


# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Use a subset of MNIST data
n_samples = 000
X_train = X_train[:n_samples].reshape(n_samples, -1)
X_train = X_train.astype(np.float32) / 255.0

# Create views using PCA and t-SNE
n_components = 3
Xs = create_views(X_train, n_components)

# Set parameters
n_clusters = 10
alpha = 0.5
sigma = 1.0

# Perform adaptive sparse graph learning for multi-view spectral clustering
labels, H_new = adaptive_sparse_graph_learning(Xs, n_clusters, alpha, sigma)


ari = adjusted_rand_score(y_train[:n_samples], labels)
print(f"Adjusted Rand Index: {ari:.4f}")


def map_labels(true_labels, predicted_labels):
    mapped_labels = np.zeros_like(predicted_labels)
    for label in np.unique(predicted_labels):
        mask = (predicted_labels == label)
        sub_true_labels = true_labels[mask]
        most_common_true_label = mode(sub_true_labels)[0][0]
        mapped_labels[mask] = most_common_true_label
    return mapped_labels


# Map the predicted cluster labels to the true labels
mapped_labels = map_labels(y_train[:n_samples], labels)

# Compute accuracy
accuracy = accuracy_score(y_train[:n_samples], mapped_labels)
print(f"Accuracy: {accuracy:.4f}")

# Compute NMI
nmi = normalized_mutual_info_score(y_train[:n_samples], labels)
print(f"Normalized Mutual Information: {nmi:.4f}")


# Visualization

# Visualization
def plot_clusters(X, labels, title):
    unique_labels = np.unique(labels)
    for label in unique_labels:
        plt.scatter(X[labels == label, 0], X[labels == label, 1], label=label)
    plt.legend()
    plt.title(title)
    plt.show()


# Apply t-SNE on the shared eigenvectors H_new
tsne = TSNE(n_components=2, random_state=42)
H_new_2d = tsne.fit_transform(H_new)

# Visualize the clustering results using the t-SNE representation of the shared eigenvectors
plot_clusters(H_new_2d, labels, "Adaptive Sparse Graph Learning Clustering")

# # Visualize the clustering results using the first two columns of the shared eigenvectors
# plot_clusters(H_new, labels, "Adaptive Sparse Graph Learning Clustering")
