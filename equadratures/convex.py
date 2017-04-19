"""A library of convex optimizers"""
import numpy as np
from scipy.linalg import det, cholesky, lstsq  

def maxdet(A, k):
    """
    Formulation of the determinant maximization as a convex program
    """
    maxiter = 30
    n_tol = 1e-3
    gap = 1.005

    # For backtracking line search parameters
    alpha = 0.01
    beta = 0.5

    # Assuming the input matrix is an np.matrix()
    m, n = A.shape
    z = np.ones((m, 1)) * float(k/m)
    g = np.zeros((m, 1))
    ones_m = np.ones((m, 1))
    ones_m_transpose = np.ones((1, m))
    kappa = np.log10(gap) * n/m

    # Objective function
    Z = np.diag(z)
    fz = -log10(np.det(A.T * Z * A)) - kappa * np.sum(np.log10(z) + np.log10(1 - z))

    # Optimization loop!
    for i in range(0, maxiter):
        W = np.inv(A.T * Z * A)
        V = A * W * A.T

        # define some z operations
        one_by_z = ones_m / z
        one_by_one_minus_z = ones_m / (ones_m - z)
        one_by_z2 = ones_m / z**2
        one_by_one_minus_z2 = ones_m / (ones_m - z)**2
        g = -np.diag(V) - kappa * (one_by_z - one_by_one_minus_z)
        H = np.multiply(V, V) + kappa * np.diag( one_by_z2 + one_by_one_minus_z2)

        # Textbook Newton's method -- compute inverse of Hessian
        R = cholesky(H)
        u = lstsq(R.T, g)
        Hinvg = lstsq(R, u)
        v = lstsq(R.T, ones_m)
        Hinv1 = lstsq(R, v)
        dz = -Hinv1 + (( ones_m_transpose * Hinvg )) / ((ones_m_transpose * Hinv1))



