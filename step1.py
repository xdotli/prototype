import numpy as np

def golden_section_search(f, a, b, tol=1e-6):
    golden_ratio = (1 + np.sqrt(5)) / 2
    c = b - (b - a) / golden_ratio
    d = a + (b - a) / golden_ratio

    while abs(b - a) > tol:
        if f(c) < f(d):
            b = d
        else:
            a = c

        c = b - (b - a) / golden_ratio
        d = a + (b - a) / golden_ratio

    return (b + a) / 2

def optimize_element(y, lam, p, alpha):
    obj = lambda z: 0.5 / alpha * (z - y)**2 + lam * abs(z)**p
    return golden_section_search(obj, -1.0, 1.0)

def admm(X, lam, p, rho, gamma, alpha, tol=1e-6, max_iter=1000):
    m, n = X.shape
    C = np.zeros((n, n))
    Z = np.zeros((n, n))
    U = np.zeros((n, n))

    for k in range(max_iter):
        # c-update
        C = np.linalg.inv(X.T @ X + rho * np.eye(n)) @ (X.T @ X + rho * Z - U)

        # z-update (proximal gradient method)
        Z_prev = Z.copy()
        for t in range(max_iter):
            grad_g = 2 * gamma * np.maximum(Z_prev - 1, 0) - rho * (Z_prev - C + U / rho)
            Y = Z_prev - alpha * grad_g
            Z_new = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    Z_new[i, j] = optimize_element(Y[i, j], lam, p, alpha)

            if np.linalg.norm(Z_new - Z_prev) < tol:
                break
            Z_prev = Z_new

        Z = Z_new

        # u-update
        U = U + rho * (Z - C)

        # Check convergence
        primal_residual = np.linalg.norm(Z - C)
        dual_residual = np.linalg.norm(rho * (Z - Z_prev))
        if primal_residual < tol and dual_residual < tol:
            break

    return C, Z, U

# Example usage
X = np.random.rand(100, 100)
lam = 0.1
p = 0.5
rho = 1.0
gamma = 1.0
alpha = 0.01

C, Z, U = admm(X, lam, p, rho, gamma, alpha)
