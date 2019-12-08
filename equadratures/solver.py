"""Solvers for computing of a linear system."""
import numpy as np
from scipy.linalg import qr
from copy import deepcopy
class Solver(object):
    """
    Returns solver functions for solving Ax=b
    :param string method: The method used for solving the linear system. Options include: ``compressed-sensing``, ``least-squares``, ``minimum-norm``, ``numerical-integration`` and ``least-squares-with-gradients``.
    :param dict solver_args: Optional arguments centered around the specific solver.
            :param numpy.ndarray noise-level: The noise-level to be used in the basis pursuit de-noising solver.
            :param bool verbose: Default value of this input is set to ``False``; when ``True`` a string is printed to the screen detailing the solver convergence and condition number of the matrix.
    """
    def __init__(self, method, solver_args):
        self.method = method
        self.solver_args = solver_args
        self.noise_level = None
        self.verbose = False
        if self.solver_args is not None:
            if 'noise-level' in self.solver_args: self.noise_level = solver_args.get('noise-level')
            if 'verbose' in self.solver_args: self.verbose = solver_args.get('verbose')
        if self.method.lower() == 'compressed-sensing' or self.method.lower() == 'compressive-sensing':
            self.solver = lambda A, b: basis_pursuit_denoising(A, b, self.noise_level, self.verbose)
        elif self.method.lower() == 'least-squares':
            self.solver = lambda A, b: least_squares(A, b, self.verbose)
        elif self.method.lower() == 'minimum-norm':
            self.solver = lambda A, b: minimum_norm(A, b)
        elif self.method.lower() == 'numerical-integration':
            self.solver = lambda A, b: orthogonal_linear_system(A, b)
        elif self.method.lower() == 'least-squares-with-gradients':
            self.solver = lambda A, b, C, d: constrained_least_squares(A, b, C, d, self.verbose)
    def get_solver(self):
        return self.solver
def least_squares(A, b, verbose):
    if np.__version__ < '1.14':
        alpha = np.linalg.lstsq(A, b)
    else:
        alpha = np.linalg.lstsq(A, b, rcond=None)
    if verbose is True:
        print('The condition number of the matrix is '+str(np.linalg.cond(A))+'.')
    return alpha[0]
def minimum_norm(A, b):
    Q, R, pvec = qr(A, pivoting=True)
    m, n = A.shape
    r = np.linalg.matrix_rank(A)
    Q1 = Q[0:r, 0:r]
    R1 = R[0:r, 0:r]
    indices = np.argsort(pvec)
    P = np.eye(n)
    temp = P[indices,:]
    P1 = temp[0:n, 0:r]
    x = np.dot(P1 ,  np.dot( np.linalg.inv(R1)  , np.dot( Q1.T , b ) ) )
    x = x.reshape(n, 1)
    return x
def orthogonal_linear_system(A, b):
    coefficients = np.dot(A.T, b)
    return coefficients
def constrained_least_squares(A, b, C, d, verbose):
    # Size of matrices!
    m, n = A.shape
    p, q = b.shape
    k, l = C.shape
    s, t = d.shape

    # Check that the number of elements in b are equivalent to the number of rows in A
    if m != p:
        raise(ValueError, 'solver: error: mismatch in sizes of A and b')
    elif k != s:
        raise(ValueError, 'solver: error: mismatch in sizes of C and d')
    if m >= n:
        return least_squares(np.vstack([A, C]), np.vstack([b, d]), verbose)
    else:
        return null_space_method(C, d, A, b, verbose)
def null_space_method(Ao, bo, Co, do, verbose):
    A = deepcopy(Ao)
    C = deepcopy(Co)
    b = deepcopy(bo)
    d = deepcopy(do)
    m, n = A.shape
    p, n = C.shape
    Q, R = np.linalg.qr(C.T, 'complete')
    Q1 = Q[0:n, 0:p]
    Q2 = Q[0:n, p:n]
    # Lower triangular matrix!
    L = R.T
    L = L[0:p, 0:p]
    y1 = least_squares(L, d, verbose)
    c = b - np.dot( np.dot(A , Q1) , y1)
    AQ2 = np.dot(A , Q2)
    y2 = least_squares(AQ2 , c, verbose)
    x = np.dot(Q1 , y1) + np.dot(Q2 , y2)
    cond = np.linalg.cond(AQ2)
    if verbose is True:
        print('The condition number of the matrix is '+str(cond)+'.')
    return x
def basis_pursuit_denoising(Ao, bo, noise_level, verbose):
    A = deepcopy(Ao)
    y = deepcopy(bo)
    N = A.shape[0]
    # Possible noise levels
    log_eta = [-8,-7,-6,-5,-4,-3,-2,-1]
    if noise_level is None:
        eta = [float(10**i) for i in log_eta]
    else:
        try:
            len(noise_level)
            eta = noise_level
        except TypeError:
            eta = [noise_level]
        log_eta =  [np.log10(i) for i in eta]
    errors = []
    mean_errors = np.zeros(len(eta))
    # 5 fold cross validation
    for e in range(len(eta)):
        for n in range(5):
            try:
                indices = [int(i) for i in n * np.ceil(N/5.0) + range(int(np.ceil(N/5.0))) if i < N]
                A_ver = A[indices]
                A_train = np.delete(A, indices, 0)
                y_ver = y[indices].flatten()
                if len(y_ver) == 0:
                    continue
                y_train = np.delete(y, indices).flatten()

                x_train = _bp_denoise(A_train, y_train, eta[e])
                y_trained = np.reshape(np.dot(A_ver, x_train), len(y_ver))

                assert y_trained.shape == y_ver.shape
                errors.append(np.mean(np.abs(y_trained - y_ver))/len(y_ver))
            except np.linalg.LinAlgError:
                continue
        if len(errors) == 0:
            mean_errors[e] = np.inf
        else:
            mean_errors[e] = np.mean(errors)
    sorted_ind = np.argsort(mean_errors)
    x = None
    ind = 0
    while x is None:
        if ind >= len(log_eta):
            raise ValueError('Singular matrix!! Reconsider sample points!')
        try:
            x = _bp_denoise(A, y, eta[sorted_ind[ind]])
        except np.linalg.LinAlgError:
            ind += 1
    if verbose:
        print('The noise level used is '+str(eta[sorted_ind[ind]])+'.')
    return np.reshape(x, (len(x),1))
def _CG_solve(A, b, max_iters, tol):
    """
    Solves Ax = b iteratively using conjugate gradient, assuming A is a symmetric positive definite matrix.
    :param numpy-matrix A:
        The matrix.
    :param numpy-array b:
        The right hand side column vector.
    :param int max_iters:
        Maximum number of iterations for the conjugate gradient algorithm.
    :param double tol:
        Tolerance for cut-off.
    """
    if not(np.all(np.linalg.eigvals(A) > 0)):
        raise ValueError('A is not symmetric positive definite.')
    n = A.shape[0]
    b = b.astype(np.float64)
    b = b.reshape((len(b),1))


    #Initialization
    x = np.zeros((n,1))
    r = b.copy()

    d = r.copy()
    iterations = 0
    delta = sum(r**2)
    delta_0 = sum(b**2)
    bestx = x.copy()
    bestres = np.sqrt(delta/delta_0)
    residual = np.sqrt(delta / delta_0)

    while (iterations < max_iters) and (delta > (tol**2) * delta_0):

        q = np.dot(A,d)
        alpha = delta / sum(d * q)

        x += alpha * d
        r -= alpha * q
        new_delta = sum(r**2)
        beta = new_delta / delta
        d = r + beta * d

        if np.sqrt(delta/delta_0) < bestres:
            bestx = x.copy()
            bestres = np.sqrt(delta/delta_0)

        delta = new_delta
        residual = np.sqrt(delta / delta_0)
        iterations += 1

    return x.flatten(), residual, iterations
def _bp_denoise(A, b, epsilon, x0 = None, lbtol = 1e-3, mu = 10, cgtol = 1e-8, cgmaxiter = 200, verbose = False, use_CG = False):
    """
    Solving the basis pursuit de-noising problem.
    :param numpy-matrix A:
        The matrix.
    :param numpy-array b:
        The right hand side column vector.
    :param double epsilon:
        The noise.
    :param numpy-array x0:
        Initial solution  if not provided the least norm solution is used.
    """
    newtontol = lbtol
    newtonmaxiter = 50

    b = b.flatten()

    # starting point --- make sure that it is feasible
    if not(x0 is None):
        if (np.linalg.norm(np.dot(A,x0) - b) > epsilon):
            if verbose:
                print('Starting point infeasible  using x0 = At*inv(AAt)*y.')
            if use_CG:
                w, cgres, cgiter =  _CG_solve(np.dot(A,A.T),b,cgmaxiter,cgtol)
            else:
                w = np.linalg.solve(np.dot(A,A.T),b).flatten()
                cgres = np.linalg.norm(np.dot(np.dot(A,A.T), w).flatten() - b.flatten()) / np.linalg.norm(b)
                cgiter = -1
            if cgres > .5:
              if verbose:
                    print("cgres = " + str(cgres) )
                    print('A*At is ill-conditioned: cannot find starting point' )
              xp = x0.copy()
              return xp
            x0 = np.dot(A.T, w)
    else:
        if verbose:
            print('No x0. Using x0 = At*inv(AAt)*y.')
        if use_CG:
            w, cgres, cgiter =  _CG_solve(np.dot(A,A.T),b,cgmaxiter,cgtol)
        else:
            w = np.linalg.solve(np.dot(A,A.T),b).flatten()
            cgres = np.linalg.norm(np.dot(np.dot(A,A.T), w).flatten() - b.flatten()) / np.linalg.norm(b)
            cgiter = -1
        if cgres > .5:
              if verbose:
                    print("cgres = " + str(cgres) )
                    print('A*At is ill-conditioned: cannot find starting point' )
        x0 = np.dot(A.T, w)

    x = x0.copy()
    r = np.reshape(np.dot(A, x), len(b)) - b
    N = len(x0)
    u = (0.95)*np.abs(x0) + (0.10)*np.max(np.abs(x0))     #arbitrary u starting point?
    if verbose:
        print('Original l1 norm = ' + str(np.sum(np.abs(x0))) + ', original functional = ' + str(np.sum(u)) )
    tau = np.max([(2.0*N+1)/np.sum(np.abs(x0)), 1.0])
    lbiter = int(np.max([np.ceil((np.log(2.0*N+1) - np.log(lbtol) - np.log(tau)) / np.log(mu)), 0.0]))
    if verbose:
        print('Number of log barrier iterations = ' + str(lbiter) )
    totaliter = 0
    for ii in range(lbiter+1):
      xp, up, ntiter =  _l1qc_newton(x, u, A, b, epsilon, tau, newtontol, newtonmaxiter, cgtol, cgmaxiter, verbose, use_CG)
      totaliter += ntiter
      if verbose:
          print('Log barrier iter = ' + str(ii) + ', l1 = ' + str(np.sum(np.abs(xp))) + ', functional = ' + str(np.sum(up)) + \
          ', tau = ' + str(tau) + ', total newton iter = ' + str(totaliter))

      x = xp.copy()
      u = up.copy()
      tau *= mu
    return xp
def _l1qc_newton(x0, u0, A, b, epsilon, tau, newtontol, newtonmaxiter, cgtol, cgmaxiter, verbose, use_CG):
    # line search parameters
    alpha = 0.01
    beta = 0.5
    AtA = np.dot(A.T,A)
    x = x0.flatten()
    u = u0.flatten()
    r = np.dot(A, x).flatten() - b.flatten()
    fu1 = x - u
    fu2 = -x - u
    fe = 0.5*(np.asscalar(np.dot(r.T,r)) - epsilon**2)
    f = np.sum(u) - (1.0/tau) * (np.sum(np.log(-fu1)) + np.sum(np.log(-fu2)) + np.log(-fe))

    niter = 0
    done = 0
    while (not(done)):

      atr = np.dot(A.T, r)

      ntgz = 1.0/fu1 - 1.0/fu2 + 1.0/fe * atr
      ntgu = -tau - 1.0/fu1 - 1.0/fu2
      gradf = - (1.0/tau) * np.hstack([ntgz, ntgu])

      sig11 = 1.0/fu1**2 + 1.0/fu2**2
      sig12 = -1.0/fu1**2 + 1.0/fu2**2
      sigx = sig11 - sig12**2/sig11

      w1p = ntgz - sig12/sig11 *ntgu

      H11p = np.diag(sigx.reshape(len(sigx))) - (1.0/fe) * AtA + (1.0/fe)**2 * np.outer(atr,atr)
      if use_CG:
          dx, cgres, cgiter =  _CG_solve(H11p, w1p, cgmaxiter, cgtol)
      else:
          dx = np.linalg.solve(H11p, w1p).flatten()
          cgres = np.linalg.norm(np.dot(H11p, dx).flatten() - w1p.flatten()) / np.linalg.norm(w1p)
          cgiter = -1
      if (cgres > 0.5):
          if verbose:
              print("cgres = " + str(cgres) )
              print('Cannot solve system.  Returning previous iterate.' )
          xp = x.flatten()
          up = u.flatten()
          return xp, up, 0
      Adx = np.dot(A,dx).flatten()


      du = (1.0/sig11) * ntgu - (sig12/sig11)*dx

      # minimum step size that stays in the interior
      aqe = np.dot(Adx.T, Adx)
      bqe = 2.0*np.dot(r.T, Adx)
      cqe = np.asscalar(np.dot(r.T,r)) - epsilon**2

      smax = np.min(np.hstack([ 1.0,np.min(np.hstack([-fu1[(dx-du) > 0] / (dx[(dx-du) > 0] - du[(dx-du) > 0]),\
        -fu2[(-dx-du) > 0] / (-dx[(-dx-du) > 0] - du[(-dx-du) > 0]), \
        np.reshape((-bqe + np.sqrt(bqe**2 - 4 * aqe * cqe)) / (2.0*aqe), (1,)) ] ))]))
      s = (0.99) * smax

      # backtracking line search
      suffdec = 0
      backiter = 0
      while not(suffdec):
        xp = x + s*dx
        up = u + s*du
        rp = r + s*Adx

        fu1p = xp - up
        fu2p = -xp - up

        fep = 0.5 * (np.linalg.norm(rp)**2 - epsilon**2)
        fp = np.sum(up) - (1.0/tau) * (np.sum(np.log(-fu1p)) + np.sum(np.log(-fu2p)) + np.log(-fep))

        flin = f + alpha * s * (np.dot(gradf.T, np.hstack([dx, du])))

        suffdec = (fp <= flin)
        s = beta * s
        backiter = backiter + 1
        if (backiter > 32):
          if verbose:
              print('Stuck on backtracking line search, returning previous iterate.')
          xp = x.copy()
          up = u.copy()
          return xp,up,niter

      # set up for next iteration
      x = xp.copy()
      u = up.copy()
      r = rp.copy()
      fu1 = fu1p.copy()
      fu2 = fu2p.copy()
      fe = fep.copy()
      f = fp.copy()

      lambda2 = -(np.dot(gradf, np.hstack([dx, du])))
      stepsize = s*np.linalg.norm(np.hstack([dx, du]))
      niter = niter + 1
      done = (lambda2/2 < newtontol) | (niter >= newtonmaxiter)
      if verbose:
          print('Newton iter = ' + str(niter) + ', Functional = ' + str(f) + ', Newton decrement = ' + str(lambda2/2) + ', Stepsize = ' + str(stepsize))
          print('                CG Res = ' + str(cgres) + ', CG Iter = ' + str(cgiter))
    return xp, up, niter
