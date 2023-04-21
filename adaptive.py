import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

# Load MNIST dataset
mnist = fetch_openml('mnist_784')
X = mnist.data.values / 255.0
y = mnist.target.astype(int).values
n_samples = 1000
indices = np.random.choice(X.shape[0], n_samples, replace=False)
X, y = X[indices], y[indices]

# PCA for dimensionality reduction
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X)


def adaptive_spectral_clustering(X, k, c, alpha=1.0, beta=1.0):
    def affinity_matrix(X, k, beta):
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(X)
        distances, indices = nbrs.kneighbors(X)
        theta = np.mean(distances[:, 1:], axis=1) / k
        dij = distances[:, 1:]
        s_hat = np.zeros((X.shape[0], X.shape[0]))
        for i in range(X.shape[0]):
            # s_hat[i, indices[i, 1:]] = (
            #     dij[i, k] - dij[i]) / (k * dij[i, k] - np.sum(dij[i]))
            s_hat[i, indices[i, 1:]] = (
                dij[i, k - 1] - dij[i]) / (k * dij[i, k - 1] - np.sum(dij[i]))
        return s_hat

    def laplacian_matrix(S):
        D = np.diag(np.sum(S, axis=1))
        L = D - S
        return L

    def spectral_embedding(L, c):
        eigvals, eigvecs = np.linalg.eigh(L)
        H = eigvecs[:, :c]
        return H

    def optimal_rotation(H, Y):
        U, _, V = np.linalg.svd(np.matmul(H.T, Y))
        R = np.matmul(U, V.T)
        return R

    def clustering(H, R, c):
        HR = np.dot(H, R)
        labels = np.argmax(HR, axis=1)
        return labels

    S = affinity_matrix(X, k, beta)
    L = laplacian_matrix(S)
    H = spectral_embedding(L, c)
    kmeans = KMeans(n_clusters=c, random_state=42).fit(H)
    Y_pred = kmeans.labels_
    Y = np.zeros((n_samples, c))
    Y[np.arange(n_samples), Y_pred] = 1
    R = optimal_rotation(H, Y)
    labels = clustering(H, R, c)
    return labels


# Hyperparameters
k = 5
c = 10

# Run the adaptive spectral clustering algorithm
predicted_labels = adaptive_spectral_clustering(X_pca, k, c)

# Adjust predicted_labels for the best match with true labels
row_ind, col_ind = linear_sum_assignment(-np.array(
    [[np.sum((predicted_labels == k) & (y == l)) for l in range(c)] for k in range(c)]))
label_map = {k: l for k, l in zip(row_ind, col_ind)}
predicted_labels = np.array([label_map[l] for l in predicted_labels])

# Calculate the accuracy
accuracy = accuracy_score(y, predicted_labels)
print(f"Accuracy: {accuracy}")
