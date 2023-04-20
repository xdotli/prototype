import numpy as np
from numpy import dot

def gradient_descent_U(X, U, V, alpha, lam):
    n = U.shape[0]
    U_new = U - alpha * (2 * dot(X.T, dot(X, dot( dot(U, V.T)-np.eye(n), V))) + lam * U)
    return U_new

def gradient_descent_V(X, U, V, alpha, lam):
    n = V.shape[0]
    V_new = V - alpha * (2 * dot(U.T, dot(X.T, dot(X, dot(U,V)))) + lam*V)
    return V_new

def normalize(X):
    return (X - np.min(X)) / (np.max(X) - np.min(X))

def admm(X, lam, alpha, tol=1e-6, max_iter=1000):
    n = X.shape[0]
    U = np.random.rand(n, n) * 0.01
    V = np.random.rand(n, n) * 0.01

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
        #print("primal_residual:", primal_residual)
        #print("dual_residual:", dual_residual)

        if primal_residual <= tol and dual_residual <= tol:
            break

    return U, V

# Example usage
X = np.random.rand(100, 100)
n = 100
#X = normalize(X)
lam = 1
alpha = 0.0001

U, V = admm(X, lam, alpha, max_iter=5000)
C = dot(U,V.T)

# Step2
A = (C+C.T)/2
D = np.linalg.norm(X)
L = np.eye(n) - dot(D**(-0.5),dot(A,D)) - 0.5

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

    # Generate a random n*c matrix
    A = np.random.rand(n, c)

    # Apply the Gram-Schmidt process to the rows of A
    H = gram_schmidt(A)

    # Verify that transpose(Q) * Q is approximately equal to the identity matrix
    assert np.allclose(H.T @ H, np.identity(c, dtype=float), atol=1e-8)

    return H


def gradient_H(H,L,R,Y,Q,alpha,rho):
    g_h = 2*L + 1 + alpha*(dot(2*H, dot(R, R.T))-2*dot(Y,R.T)) + dot(H, Q+Q.T) + 2*rho*(dot(H,dot(H.T,H) - 2*H))
    return g_h

def gradient_R(H,R,Y,P,alpha,rho):
    g_r = alpha*(2*dot(H.T,dot(H,R))-2*dot(H.T,Y)) + dot(R, P+P.T) + 2*rho*(dot(R,dot(R.T,R))-2*R)
    return g_r

def h_function(H,R,c):
    hr = dot(H,R)
    ind = hr.argmax(axis=1)
    Y = np.zeros((n,c))
    for i in range(n):
        Y[i][ind[i]] = 1
    return Y


def step_2_admm(c, n, rho,alpha,tau,tol = 1e-6, max_ier = 1000):
    H = find_h(n,c)# n*c
    R = np.eye(c) # c*c
    Y = find_h(n,c) # n*c
    P = np.zeros((c,c))
    Q = np.zeros((c,c))

    for k in range(max_ier):
        # update H
        grad_h = gradient_H(H,L,R,Y,Q,alpha,rho)
        H = H - tau*grad_h
        #print(H)

        # update R
        grad_r = gradient_R(H,R,Y,P,alpha,rho)
        R = R - tau*grad_r
        #print(R)

        # update Y
        Y = h_function(H,R,n)

        P = P + rho*(dot(R.T,R)-np.eye(c))
        Q = Q + rho*(dot(H.T,H)-np.eye(c))

        # check convergence
        if np.linalg.norm(grad_h) < tol and np.linalg.norm(grad_r) < tol:
            break
    
    return Y



    
Y = step_2_admm(10,100, rho = 1,alpha = 0.001,tau = 0.0001,tol = 1e-6, max_ier = 1000)
print(Y)