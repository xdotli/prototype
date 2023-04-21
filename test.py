import tensorflow as tf
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler

mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess the dataset
X_train = X_train.reshape(60000,-1)
print(X_train.shape)
print(X_train[0])