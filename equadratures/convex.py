"""A library of convex optimizers"""
import numpy as np
from scipy.linalg import det, cholesky, lstsq  

def maxdet(A, k):
    """
    Formulation of the determinant maximization as a convex program
    """
    maxiter = 50
    n_tol = 1e-12
    gap = 1.005

    # For backtracking line search parameters
    alpha = 0.01
    beta = 0.5

    # Assuming the input matrix is an np.matrix()
    m, n = A.shape
    if m < n:
        raise(ValueError, 'maxdet(): requires the number of columns to be greater than the number of rows!')
    z = np.ones((m, 1)) * float(k)/float(m)
    g = np.zeros((m, 1))
    ones_m = np.ones((m, 1))
    ones_m_transpose = np.ones((1, m))
    kappa = np.log(gap) * n/m

    # Objective function
    Z = diag(z)
    fz = -np.log(np.linalg.det(A.T * Z * A)) - kappa * np.sum(np.log(z) + np.log(1.0 - z))

    #print 'Iteration \t Step size \t Newton decrement \t Objective \t log_det'
    #print str(0)+'\t'+'--'+'\t'+'--'+'\t'+str(-fz)+'\t'+str(np.log(np.linalg.det(A.T * Z * A)) )
        
    # Optimization loop!
    for i in range(0, maxiter) :
        Z = diag(z)
        W = np.linalg.inv(A.T * Z * A)
        V = A * W * A.T
        vo = np.matrix(np.diag(V))
        vo = vo.T

        # define some z operations
        one_by_z = ones_m / z
        one_by_one_minus_z = ones_m / (ones_m - z)
        one_by_z2 = ones_m / z**2
        one_by_one_minus_z2 = ones_m / (ones_m - z)**2
        g = -vo- kappa * (one_by_z - one_by_one_minus_z)
        H = np.multiply(V, V) + kappa * diag( one_by_z2 + one_by_one_minus_z2)

        # Textbook Newton's method -- compute inverse of Hessian
        R = np.matrix(cholesky(H) )
        u = lstsq(R.T, g)
        Hinvg = lstsq(R, u[0])
        Hinvg = Hinvg[0]
        v = lstsq(R.T, ones_m)
        Hinv1 = lstsq(R, v[0])
        Hinv1 = Hinv1[0]
        dz = -Hinvg + (np.dot( ones_m_transpose , Hinvg ) / np.dot(ones_m_transpose , Hinv1)) * Hinv1


        deczi = indices(dz, lambda x: x < 0)
        inczi = indices(dz, lambda x: x > 0)
        a1 = 0.99* -z[deczi, 0] / dz[deczi, 0]
        a2 = (1 - z[inczi, 0] )/dz[inczi, 0]  
        s = np.min(np.vstack([1.0, np.vstack(a1), np.vstack(a2) ] )  )
        flag = 1

        while flag == 1:
            zp = z + s*dz
            Zp = diag(zp)
            fzp = -np.log(np.linalg.det(A.T * Zp * A) ) - kappa * np.sum(np.log(zp) + np.log(1 - zp)  )
            const = fz + alpha * s * g.T * dz
            if fzp <= const[0,0]:
                flag = 2
            if flag != 2:
                s = beta * s
        z = zp
        fz = fzp
        sig = -g.T * dz * 0.5
        #print str(i+1)+'\t'+str(s)+'\t'+str(sig[0,0])+'\t'+str(-fz)+'\t'+str(np.log(np.linalg.det(A.T * diag(z) * A)) )
        if( sig[0,0] <= n_tol):
            break
        zsort = np.sort(z, axis=0)
        thres = zsort[m - k - 1]
        zhat, not_used = find(z, thres)
    
    zsort = np.sort(z, axis=0)
    thres = zsort[m - k - 1]
    zhat, not_used = find(z, thres)
    p, q = zhat.shape
    Zhat = diag(zhat)
    L = np.log(np.linalg.det(A.T * Zhat  * A)) 
    ztilde  = z
    Utilde = np.log(np.linalg.det(A.T * diag(z) * A))  + 2 * m * kappa

    return zhat, L, ztilde, Utilde

def CG_solve(A, b, max_iters, tol):
    """
    Solves Ax = b iteratively using conjugate gradient.
    A must be a SPD matrix
    """
    if not(np.all(np.linalg.eigvals(A) > 0)):
        raise ValueError('A is not symmetric positive definite.')
    n = A.shape[0]
    b = b.astype(np.float64)
    b = b.reshape((len(b),1))
    
    
    #Initialization
    x = np.zeros((n,1))
    r = b.copy()
    p = r.copy()
    iterations = 0
    delta = sum(r**2)
    delta_0 = sum(b**2)
    residual = np.sqrt(delta / delta_0)
    
    while (iterations < max_iters) and (delta > (tol**2) * delta_0):
        alpha = delta / sum(r * np.dot(A,r))
        x += alpha * p
        r -= alpha * np.dot(A,p)
        new_delta = sum(r**2)
        beta = new_delta / delta
        p = r + beta * p
        residual = np.sqrt(delta / delta_0)
        delta = new_delta
        iterations += 1
    
    
    return x, residual, iterations
    
def nn_min_l1(A, b, x0 = None, cgtol = None, pdtol = None, pdmaxiter = None):
    """
    l1 minimization with equality constraint Ax = b.
    (Noiseless basis pursuit)
    Adapted from l1-magic (Candes and Romberg)
    """
    
    # Free parameters
    alpha = .01
    beta = .5
    mu = 10
    
    if cgtol is None:
        cgtol = 1e-8
    if pdtol is None:
        pdtol = 1e-3
    if pdmaxiter is None:
        pdmaxiter = 50
    
    # Find initial solution, if none provided or provided but infeasible
    if not(x0 is None):
        if np.linalg.norm(np.dot(A,x0) - b)/np.linalg.norm(b) > cgtol:
            # Infeasible, use A^t (AA^t)^-1 b (min l2 norm) instead
            q, cgres, _ = CG_solve(np.dot(A,A.T), b)
            if cgres > 0.5:
                raise ValueError('AA^T is too ill conditioned. Cannot find starting point.')
            x0 = np.dot(A.T, q)
    else:
        q, cgres, _ = CG_solve(np.dot(A,A.T), b)
        if cgres > 0.5:
            raise ValueError('AA^T is too ill conditioned. Cannot find starting point.')
        x0 = np.dot(A.T, q)
    
    gradf0 = np.vstack(np.zeros((N,1),dtype = np.float64) np.ones((N,1),dtype = np.float64))
    N = len(x0)
    x = x0.copy()
    x = x.reshape((len(x),1))
    u = 0.95*np.abs(x0) + 0.10*np.max(np.abs(x0));
    u = u.reshape((len(u),1))
    
    # First iteration
    fu1 = x - u
    fu2 = -x - u
    #initialize lambda to be 1/f
    lam_u1 = 1.0 / fu1
    lam_u2 = 1.0 / fu2
    #initialize nu (v) to be -A*(lam_u1 - lam_u2)
    v = np.dot(-A, (lam_u1-lam_u2))
    
    sdg = -(sum(fu1*lamu1) + sum(fu2*lamu2))
    tau = mu * 2 * N / sdg
    
    #Calculate the residuals
    Atv = np.dot(A.T, v)
    r_pri = np.dot(A, x) - b
    r_cent = np.vstack(-lam_u1 * fu1, -lam_u2 * fu2) - 1.0/tau
    r_dual = gradf0 + np.vstack(lam_u1 - lam_u2, -lam_u1 - lam_u2) + np.vstack(Atv, np.zeros((N,1)))
    resnorm = np.linalg.norm(np.vstack((r_pri, r_cent, r_dual)))
    
    pditer = 0
    done = (sdg < pdtol) or (pditer >= pdmaxiter)
    
    while not(done):
        pditer += 1
        
        # Solve for delta_nu
        
        # Then get delta_x
        
        # Then get delta_u
        
        # Then get delta_lambda for both u1 and u2
        
        # Then get step size:
        # 1. Must be in interior
        # 2. Must decrease resnorm sufficiently
        
        # Prepare for next iteration

def binary2indices(zhat):
    """
    Simple utility that converts a binary array into one with indices!
    """
    pvec = []
    m, n = zhat.shape
    for i in range(0, m):
        if(zhat[i,0] == 1):
            pvec.append(i)
    return pvec
            
def indices(a, func): 
    return [i for (i, val) in enumerate(a) if func(val)]

def diag(vec):
    m = len(vec)
    D = np.zeros((m, m))
    for i in range(0, m):
        D[i,i] = vec[i,0]
    return D

def find(vec, thres):
    t = []
    vec_new = []
    for i in range(0, len(vec)):
        if vec[i] > thres:
            t.append(i)
            vec_new.append(1.0)
        else:
            vec_new.append(0.0)
    vec_new = np.matrix(vec_new)
    vec_new = vec_new.T
    return vec_new, t

