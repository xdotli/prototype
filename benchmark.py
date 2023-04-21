import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment

# Load the MNIST dataset
digits = load_digits()
data = digits.data
targets = digits.target

# Use a subset of the MNIST dataset
subset_size = 1000
indices = np.random.choice(len(data), subset_size, replace=False)
data_subset = data[indices]
targets_subset = targets[indices]

# Preprocess the data
scaler = StandardScaler()
data_subset_scaled = scaler.fit_transform(data_subset)

# Apply the SpectralClustering algorithm
num_clusters = 10
spectral_clustering = SpectralClustering(
    n_clusters=num_clusters, n_neighbors=10)
labels = spectral_clustering.fit_predict(data_subset_scaled)

# Calculate the clustering accuracy


def clustering_accuracy(y_true, y_pred):
    cost_matrix = -np.array([[np.sum((y_pred == k) & (y_true == l))
                            for l in range(max(y_true) + 1)] for k in range(max(y_pred) + 1)])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    label_map = {k: l for k, l in zip(row_ind, col_ind)}

    # Check if all labels in y_pred exist in label_map
    if not all(label in label_map for label in set(y_pred)):
        return 0

    y_pred_adjusted = np.array([label_map[label] for label in y_pred])
    return np.mean(y_true == y_pred_adjusted)


acc = clustering_accuracy(targets_subset, labels)
print("Accuracy:", acc)

# Calculate the Normalized Mutual Information (NMI) as a performance metric
nmi = normalized_mutual_info_score(targets_subset, labels)
print("Normalized Mutual Information:", nmi)

# Calculate the Adjusted Rand Index (ARI) as a performance metric
ari = adjusted_rand_score(targets_subset, labels)
print("Adjusted Rand Index:", ari)
