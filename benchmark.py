import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

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

# Calculate the Adjusted Rand Index (ARI) as a performance metric
acc = accuracy_score(targets_subset, labels)
print("Accuracy:", acc)
