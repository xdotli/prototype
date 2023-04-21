import numpy as np
from numpy import dot

def gradient_descent_U(X, U, V, alpha, lam):
    n = U.shape[0]
    U_new = U - alpha * (2 * dot(X.T, dot(X, dot( dot(U, V.T)-np.eye(n), V))) + lam * U)
    return U_new

def gradient_descent_V(X, U, V, alpha, lam):

    V_new = V - alpha * (2 * dot(U.T, dot(X.T, dot(X, dot(U,V)))) + lam*V)
    return V_new

def normalize(X):
    return (X - np.min(X)) / (np.max(X) - np.min(X))

def admm(X, lam, alpha, tol=1e-6, max_iter=1000):
    n = X.shape[0]
    U = np.random.randint(low = 0, high = 100, size = (n, n)) *0.01
    V = np.random.randint(low = 0, high = 100, size = (n, n)) *0.01

    for k in range(max_iter):
        # u-update
        U_prev = U
        U = gradient_descent_U(X, U_prev, V, alpha, lam)

        # v-update
        V_prev = V
        V = gradient_descent_V(X, V_prev, U, alpha, lam)

        # Check convergence
        primal_residual = np.linalg.norm(U - U_prev)
        dual_residual = np.linalg.norm(V - V_prev)
        print("primal_residual:", primal_residual)
        print("dual_residual:", dual_residual)

        if primal_residual <= tol and dual_residual <= tol:
            break

    return U, V

# Example usage


import numpy as np

def gram_schmidt(vectors):
    basis = []
    for v in vectors:
        w = v - sum(np.dot(v, b) * b for b in basis)
        if (w > 1e-10).any():  
            basis.append(w / np.linalg.norm(w))
    return np.array(basis)

def find_h(n, c):
    if n < c:
        raise ValueError("n must be greater than or equal to c")

    # Generate a random n*n matrix
    A = np.random.rand(n, n)

    # Apply the Gram-Schmidt process to the rows of A
    Q = gram_schmidt(A)

    # Select the first c columns from Q
    H = Q[:, :c]

    # Verify that transpose(H) * H is approximately equal to the identity matrix
    assert np.allclose(H.T @ H, np.identity(c, dtype=float), atol=1e-8)

    return H



def gradient_H(H,L,R,Y,Q,alpha,rho):
    g_h = 2*dot(L,H) + alpha*(dot(2*H, dot(R, R.T))-2*dot(Y,R.T)) + dot(H, Q+Q.T) + 2*rho*(dot(H,dot(H.T,H))-2*H)
    return g_h

def gradient_R(H,R,Y,P,alpha,rho):
    g_r = alpha*(2*dot(H.T,dot(H,R))-2*dot(H.T,Y)) + dot(R, P+P.T) + 2*rho*(dot(R,dot(R.T,R))-2*R)
    return g_r

def h_function(H,R,n,c):
    hr = dot(H,R)
    ind = hr.argmax(axis=1)
    Y = np.zeros((n,c))
    for i in range(n):
        Y[i][ind[i]] = 1
    return Y


def step_2_admm(c, n, rho,alpha,tau_1,tau_2,tol = 1e-6, max_ier = 1000):
    H = find_h(n,c)# n*c
    R = np.eye(c) # c*c
    Y = find_h(n,c) # n*c
    P = np.zeros((c,c))
    Q = np.zeros((c,c))

    for k in range(max_ier):
        # update H
        grad_h = gradient_H(H,L,R,Y,Q,alpha,rho)
        H = H - tau_1*grad_h
        #print(H)

        # update R
        grad_r = gradient_R(H,R,Y,P,alpha,rho)
        R = R - tau_2*grad_r
        #print(R)

        # update Y
        Y = h_function(H,R,n,c)

        P = P + rho*(dot(R.T,R)-np.eye(c))
        Q = Q + rho*(dot(H.T,H)-np.eye(c))

        print("grad h: ",  np.linalg.norm(grad_h))
        print("grad r: ", np.linalg.norm(grad_r))
        # check convergence
        if np.linalg.norm(grad_h) < tol and np.linalg.norm(grad_r) < tol:
            print(np.linalg.norm(grad_h))
            break
    
    return Y




    
# Y = step_2_admm(10,100, rho = 1,alpha = 0.001,tau_1 = 0.0001, tau_2 = 0.00005,tol = 1e-6, max_ier = 5000)
# print(Y)

import tensorflow as tf
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess the dataset
scaler = MinMaxScaler()
X_train = X_train.reshape(60000,-1)
X_train[np.where(X_train==0)]= 1

# Use a subset of the dataset
subset_size = 784
X_train_subset = X_train[:subset_size]
y_train_subset = y_train[:subset_size]

# X = np.random.rand(100, 100)
X = X_train_subset

# Apply the provided algorithm
n = X.shape[0]
c = len(np.unique(y_train_subset))

# n = 100
#X = normalize(X)
lam = 1
alpha = 0.00000001

U, V = admm(X, lam, alpha, max_iter=2)
C = dot(U,V.T)
print("x:", X[:10][:10])
print("C:",C)

# Step2
A = (C+C.T)/2
D = np.linalg.norm(X)
L = np.eye(n) - dot(D**(-0.5),dot(A,D**(-0.5)))
print(L)



# Replace 'step_2_admm' function call with the one in your code
Y_pred = step_2_admm(c, n, rho=1, alpha=0.001, tau_1=0.0001, tau_2=0.00005, tol=1e-6, max_ier=50)

# Convert Y_pred to a 1D array of labels
y_pred = np.argmax(Y_pred, axis=1)

# Evaluate the performance
ari = adjusted_rand_score(y_train_subset, y_pred)
nmi = normalized_mutual_info_score(y_train_subset, y_pred)
acc = accuracy_score(y_train_subset, y_pred)

print(f"ARI: {ari:.4f}")
print(f"NMI: {nmi:.4f}")
print(f"ACC: {acc:.4f}")
