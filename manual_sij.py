import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt

def binary_spectral_clustering(W,plot_clustering=False):
    """
    Binary Spectral Clustering

    Args:
        W: nxn weight matrix 
        plot_clustering: Whether to scatter plot clustering
        
    Returns:
        Numpy array of labels obtained by binary spectral clustering
    """
    #Graph Laplacian
    n = W.shape[0]
    d = W@np.ones(n)
    L = sparse.spdiags(d,0,n,n) - W  

    #Find Fiedler vector
    vals, vec = sparse.linalg.eigsh(L,k=2,which='SM')
    v = vec[:,1]

    #Cluster labels
    labels = v > 0

    #Plot clustering
    if plot_clustering:
        plt.figure()
        plt.scatter(X[:,0],X[:,1],c=labels)
        plt.title('Spectral Clustering')
    
    return labels


def kmeans(X,k,plot_clustering=False,T=20):
    """
    k-means Clustering

    Args:
        X: nxm array of data, each row is a datapoint
        k: Number of clusters
        plot_clustering: Whether to plot final clustering
        T: Max number of iterations
        
    Returns:
        Numpy array of labels obtained by binary k-means clustering
    """

    #Number of data points
    n = X.shape[0]

    #Randomly choose initial cluster means
    means = X[np.random.choice(n,size=k,replace=False),:]
  
    #Initialize arrays for distances and labels
    dist = np.zeros((k,n))
    labels = np.zeros((n,))

    #Main iteration for kmeans
    num_changed = 1
    i=0
    while i < T and num_changed > 0:
    
        #Update labels 
        old_labels = labels.copy()
        for j in range(k):
            dist[j,:] = np.sum((X - means[j,:])**2,axis=1)
        labels = np.argmin(dist,axis=0)
        num_changed = np.sum(labels != old_labels)

        #Update means 
        for j in range(k):
            means[j,:] = np.mean(X[labels==j,:],axis=0)

        #Iterate counter
        i+=1

        #Plot result (red points are labels)
    if plot_clustering:
        plt.scatter(X[:,0],X[:,1], c=labels)
        plt.scatter(means[:,0],means[:,1], c='r')
        plt.title('K-means clustering')

    return labels

# import dataset MNIST
from keras.datasets import mnist
(train_X, train_y), (test_X, test_y) = mnist.load_data()
print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))
# preprocessing
import graphlearning as gl
import numpy as np

#Binary clustering problem witih 2 digits
class1 = 5
class2 = 8

#Subset data to two digits
I = train_y == class1  
J = train_y == class2
X = train_X[I | J,:]
L = train_y[I | J]

#Convert labels to 0/1
I = L == class1
L[I] = 0
L[~I] = 1

#Build Graph (sparse 10-nearest neighbor graph)
W = gl.weightmatrix.knn(X,10)

#Spectral Clustering
spectral_labels = binary_spectral_clustering(W) 
acc1 = np.mean(spectral_labels == L)
acc2 = np.mean(spectral_labels != L)
print('Spectral clustering accuracy = %.2f%%'%(100*max(acc1,acc2)))

#k-means clustering
kmeans_labels = kmeans(X,2)
acc1 = np.mean(kmeans_labels == L)
acc2 = np.mean(kmeans_labels != L)
print('K-means clustering accuracy = %.2f%%'%(100*max(acc1,acc2)))

#Show images from each cluster
gl.utils.image_grid(X[spectral_labels==0,:], n_rows=10, n_cols=10, title='Cluster 1', fontsize=26)
gl.utils.image_grid(X[spectral_labels==1,:], n_rows=10, n_cols=10, title='Cluster 2', fontsize=26)

