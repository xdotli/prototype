import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import make_scorer
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.model_selection import GridSearchCV

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


class AdaptiveSpectralClustering(BaseEstimator, ClusterMixin):
    def __init__(self, k=5, c=10, beta=1.0):
        self.k = k
        self.c = c
        self.beta = beta

    # Add a dummy function to adhere to scikit-learn's API
    def fit(self, X, y=None):
        self.labels_ = self._adaptive_spectral_clustering(X)
        return self

    def _adaptive_spectral_clustering(self, X):
        # Add the functions from the previous code here (affinity_matrix, laplacian_matrix, spectral_embedding, optimal_rotation, and clustering)

        S = affinity_matrix(X, self.k, self.beta)
        L = laplacian_matrix(S)
        H = spectral_embedding(L, self.c)
        kmeans = KMeans(n_clusters=self.c, random_state=42).fit(H)
        Y_pred = kmeans.labels_
        Y = np.zeros((n_samples, self.c))
        Y[np.arange(n_samples), Y_pred] = 1
        R = optimal_rotation(H, Y)
        labels = clustering(H, R, self.c)
        return labels

# Custom scoring function


def clustering_accuracy_score(y_true, y_pred):
    row_ind, col_ind = linear_sum_assignment(-np.array([[np.sum((y_pred == k) & (
        y_true == l)) for l in range(max(y_true) + 1)] for k in range(max(y_pred) + 1)]))
    label_map = {k: l for k, l in zip(row_ind, col_ind)}
    y_pred_adjusted = np.array([label_map[label] for label in y_pred])
    return np.mean(y_true == y_pred_adjusted)


# Instantiate the custom estimator
estimator = AdaptiveSpectralClustering()

# Set up the parameter grid for the grid search
param_grid = {
    'k': [3, 5, 7],
    'c': [8, 10, 12],
    'beta': [0.5, 1.0, 1.5]
}

# Define a custom scoring function for GridSearchCV
custom_scoring = make_scorer(clustering_accuracy_score)

# Run the grid search
grid_search = GridSearchCV(estimator, param_grid, scoring=custom_scoring, cv=5)
grid_search.fit(X_pca, y)

# Print the best set of hyperparameters
print("Best hyperparameters found by grid search:")
print(grid_search.best_params_)
