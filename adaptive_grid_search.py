from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import make_scorer, normalized_mutual_info_score, adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.model_selection import GridSearchCV

# Load MNIST dataset
mnist = fetch_openml('mnist_784')
X = mnist.data.values / 255.0
y = mnist.target.astype(int).values
n_samples = 2000
indices = np.random.choice(X.shape[0], n_samples, replace=False)
X, y = X[indices], y[indices]

# PCA for dimensionality reduction
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X)


class AdaptiveSpectralClustering(BaseEstimator, ClusterMixin):
    def __init__(self, k=5, c=10, beta=1.0):
        self.k = k
        self.c = c
        self.beta = beta

    # Add a dummy function to adhere to scikit-learn's API
    def fit(self, X, y=None):
        # self.labels_ = self._adaptive_spectral_clustering(X)
        self.labels_, self.H_transformed_ = self._adaptive_spectral_clustering(
            X)
        return self

    def predict(self, X, y=None):
        return self.fit(X, y).labels_

    def _adaptive_spectral_clustering(self, X):
        # Add the functions from the previous code here (affinity_matrix, laplacian_matrix, spectral_embedding, optimal_rotation, and clustering)
        def affinity_matrix(X, k, beta):
            nbrs = NearestNeighbors(
                n_neighbors=k+1, algorithm='ball_tree').fit(X)
            distances, indices = nbrs.kneighbors(X)
            theta = np.mean(distances[:, 1:], axis=1) / k
            dij = distances[:, 1:]
            s_hat = np.zeros((X.shape[0], X.shape[0]))
            for i in range(X.shape[0]):
                s_i = s_hat[i].copy()
                s_i[indices[i, 1:]] = (
                    dij[i, k - 1] - dij[i]) / (k * dij[i, k - 1] - np.sum(dij[i]))
                s_hat[i] = s_i
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
            U, _, V = np.linalg.svd(np.matmul(H.T, Y[:H.shape[0], :]))
            R = np.matmul(U, V.T)
            return R

        def clustering(H, R, c):
            HR = np.dot(H, R)
            labels = np.argmax(HR, axis=1)
            return labels

        S = affinity_matrix(X, self.k, self.beta)
        L = laplacian_matrix(S)
        H = spectral_embedding(L, self.c)
        kmeans = KMeans(n_clusters=self.c, random_state=42).fit(H)
        Y_pred = kmeans.labels_
        Y = np.zeros((n_samples, self.c))
        Y[np.arange(X.shape[0]), Y_pred] = 1
        R = optimal_rotation(H, Y)
        labels = clustering(H, R, self.c)
        H_transformed = np.dot(H, R)
        return labels, H_transformed

# Custom scoring function


def clustering_accuracy_score(y_true, y_pred):
    row_ind, col_ind = linear_sum_assignment(-np.array([[np.sum((y_pred == k) & (
        y_true == l)) for l in range(max(y_true) + 1)] for k in range(max(y_pred) + 1)]))
    label_map = {k: l for k, l in zip(row_ind, col_ind)}

    # Check if all labels in y_pred exist in label_map
    if not all(label in label_map for label in set(y_pred)):
        return 0

    y_pred_adjusted = np.array([label_map[label] for label in y_pred])
    return np.mean(y_true == y_pred_adjusted)


# Instantiate the custom estimator
estimator = AdaptiveSpectralClustering()

# Set up the parameter grid for the grid search
param_grid = {
    'k': [3, 5, 7, 9],
    'c': [10],
    'beta': [0.5, 1.0, 1.5, 2.0]
}

# Define a custom scoring function for GridSearchCV
custom_scoring = make_scorer(clustering_accuracy_score)

# Run the grid search
grid_search = GridSearchCV(estimator, param_grid, scoring=custom_scoring, cv=5)
grid_search.fit(X_pca, y)

# Print the best set of hyperparameters
print("Best hyperparameters found by grid search:")
print(grid_search.best_params_)

print("Accuracy, NMI, and ARI for each group of hyperparameters:")
for params, mean_score, scores in zip(grid_search.cv_results_['params'],
                                      grid_search.cv_results_[
                                          'mean_test_score'],
                                      zip(*[grid_search.cv_results_[f'split{i}_test_score'] for i in range(5)])):
    y_pred = grid_search.best_estimator_.predict(X_pca)
    nmi = normalized_mutual_info_score(y, y_pred)
    ari = adjusted_rand_score(y, y_pred)
    print("Hyperparameters: {}, Accuracy: {:.4f}, NMI: {:.4f}, ARI: {:.4f}".format(
        params, mean_score, nmi, ari))


# Instantiate the best estimator with the best hyperparameters
best_estimator = AdaptiveSpectralClustering(**grid_search.best_params_)
best_estimator.fit(X_pca)
y_pred = best_estimator.labels_
H_transformed = best_estimator.H_transformed_

# Perform t-SNE dimensionality reduction
tsne = TSNE(n_components=2, init='pca', perplexity=50,
            learning_rate=300, random_state=42)
X_tsne_transformed = tsne.fit_transform(H_transformed)

# Plot the t-SNE visualization
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne_transformed[:, 0], X_tsne_transformed[:, 1],
                      c=y_pred, cmap='viridis', alpha=0.7, s=50)
plt.title("t-SNE visualization of the clustering result")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")

# Add a legend for the clusters
legend1 = plt.legend(*scatter.legend_elements(),
                     loc="best", title="Clusters")
plt.gca().add_artist(legend1)

plt.show()
