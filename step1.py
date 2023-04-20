import numpy as np
from sympy import symbols, Eq, solve

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

def solver_z(z, u, c, lambda_, p, gamma, rho):
    n = z.shape[0]
    z_aim = symbols('z_aim')
    z_new = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if z[i][j] > 1:
                equation = Eq(lambda_ * p * z_aim ** (p-1) + gamma + (1/rho) * (z_aim - c[i][j]) + u[i][j],0)
            else:
                equation = Eq(lambda_ * p * z_aim ** (p-1) + (1/rho) * (z_aim - c[i][j]) + u[i][j],0)
            sol = solve(equation)
            z_new[i][j] = sol[0]
            #print(z_new)
    return z_new



def admm(X, lam, p, rho, gamma, alpha, tol=1e-6, max_iter=1000):
    m, n = X.shape
    C = np.zeros((n, n))
    Z = np.zeros((n, n))
    U = np.zeros((n, n))

    for k in range(max_iter):
        # c-update
        C = np.linalg.inv(2*X.T@X + (1/rho)*np.eye(n)) @ (2*X.T@X + U + (1/rho)*Z)

        # z-update
        Z_prev = Z
        Z = solver_z(Z_prev, U, C, lam, p, gamma, rho)

        # u-update
        U = U + rho * (Z - C)

        # Check convergence
        primal_residual = np.linalg.norm(C - Z)
        dual_residual = np.linalg.norm(rho * (Z - Z_prev))
        print("primal residual:",primal_residual)
        print("dual_residual:",dual_residual)
        if primal_residual <= tol and dual_residual <= tol:
            break

    return C, Z, U

# Example usage
X = np.random.rand(100, 100)
lam = 1
p = 0.8
rho = 1.0
gamma = 1.0
alpha = 0.001

C, Z, U = admm(X, lam, p, rho, gamma, alpha,max_iter=10)