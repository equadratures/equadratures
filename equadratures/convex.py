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
    fz = -np.log10(np.linalg.det(A.T * Z * A)) - kappa * np.sum(np.log10(z) + np.log10(1 - z))

    print 'Iteration \t Step size \t Newton decrement \t Objective \t log_det'
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

        deczi = indices(dz, lambda x: x < 0)
        inczi = indices(dz, lambda x: x > 0)
        s = np.min(np.vstack(1, 0.99*[-z[deczi, 0] / dz[deczi, 0] , (1 - z[inczi, 0] )/dz[inczi, 0]  ]))
        flag = 1

        while flag == 1:
            zp = z + s*dz
            fzp = -np.log10(np.linalg.det(A.T * np.diag(zp) * A) ) - kappa * np.sum(np.log10(zp) + log(1 - zp)  )

            if fzp <= fz + alpha * s * g.T * dz:
                flag == 2
            
            s = beta * s
        z = zp
        fz = fzp
        if( -g.T * dz * 0.5 <= n_tol):
            break
        zsort = np.sort(z)
        thres = zsort[m - k]
        zhat = indices(z, lambda x: z > thres)
    
    zsort = np.sort(z)
    thres = zsort[m - k]
    zhat = indices(z, lambda x: z > thres)
    L = np.log10(np.linalg.det(A.T * np.diag(zhat) * A)) 
    ztilde  = z
    Utilde = np.log10(np.linalg.det(A.T * np.diag(z) * A))  + 2 * m * kappa

    return zhat, L, ztilde, Utilde

def indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]
