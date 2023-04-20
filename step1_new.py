import numpy as np


def gradient_descent_U(X,U,V,alpha,lam):
    n = U.shape[0]
    U_new = U - alpha * 2 * X.T @ X @ (U @ V.T - np.eye(n)) @ V + lam * U
    return U_new

def gradient_descent_V(X,U,V,alpha,lam):
    V_new = V - alpha * 2 * U.T @ X.T @ X @ U @ V + lam * V
    return V_new



def admm(X, lam, alpha, tol=1e-6, max_iter=1000):
    n = X.shape[0]
    U = np.ones((n, n))
    V = np.ones((n,n))

    for k in range(max_iter):
        # u-update
        U_prev = U
        U = gradient_descent_U(X,U_prev,V,alpha,lam)

        # v-update
        V_prev = V
        V = gradient_descent_V(X,V_prev,U,alpha,lam)


        # Check convergence
        primal_residual = np.linalg.norm(U - U_prev)
        dual_residual = np.linalg.norm(V - V_prev)
        print(primal_residual)
        print(dual_residual)

        if primal_residual <= tol and dual_residual <= tol:
            break

    return U,V

# Example usage
X = np.random.rand(100, 100)
lam = 1
p = 0.8
rho = 1.0
gamma = 1.0
alpha = 0.01

U,V = admm(X, lam,alpha,max_iter=10)
print(U,V)