"""Solvers for computing of a linear system."""
import numpy as np
from scipy.linalg import qr
from scipy.optimize import linprog, minimize
from scipy.special import huber as huber_loss
from copy import deepcopy
try:
    import cvxpy as cv
    cvxpy = True
except ImportError as e:
    cvxpy = False

class Solver(object):
    """
    Returns solver functions for solving Ax=b
    :param string method: The method used for solving the linear system. Options include: ``compressed-sensing``, ``least-squares``, ``minimum-norm``, ``numerical-integration``, ``least-squares-with-gradients``, ``least-absolute-residual``, ``huber``, ``elastic-net``, ``lasso-lars`` and ``relevance-vector-machine``.
    :param dict solver_args: Optional arguments centered around the specific solver.
            :param numpy.ndarray noise-level: The noise-level to be used in the basis pursuit de-noising solver.
            :param bool verbose: Default value of this input is set to ``False``; when ``True`` a string is printed to the screen detailing the solver convergence and condition number of the matrix.
    """
    def __init__(self, method, solver_args):
        self.method = method
        self.solver_args = solver_args
        self.noise_level = None
        self.param1 = None
        self.param2 = None
        self.verbose = False
        self.max_iter = None
        self.ic = 'AIC'
        self.opt = 'osqp'
        if self.solver_args is not None:
            if 'noise-level' in self.solver_args: self.noise_level = solver_args.get('noise-level')
            if 'param1' in self.solver_args: self.param1 = solver_args.get('param1')
            if 'param2' in self.solver_args: self.param2 = solver_args.get('param2')
            if 'verbose' in self.solver_args: self.verbose = solver_args.get('verbose')
            if 'max-iter' in self.solver_args: self.max_iter = solver_args.get('max-iter')
            if 'optimiser' in self.solver_args: self.opt = solver_args.get('optimiser')
            if 'IC' in self.solver_args: self.ic = solver_args.get('IC')
        if self.opt=='osqp' and not cvxpy: 
            self.opt='scipy'
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
        elif self.method.lower() == 'least-absolute-residual':
            self.solver = lambda A, b: least_absolute_residual(A, b, self.verbose, self.opt)
        elif self.method.lower() == 'huber':
            self.solver = lambda A, b: huber(A, b, self.verbose, self.param1, self.opt)
        elif self.method.lower() == 'elastic-net':
            self.solver = lambda A, b: elastic_net(A, b, self.verbose, self.param1, self.param2, self.opt)
        elif self.method.lower() == 'lasso-lars':
            self.solver = lambda A, b: lasso(A, b, self.verbose, self.max_iter, self.ic)
        elif self.method.lower() == 'relevance-vector-machine':
            self.solver = lambda A, b: rvm(A, b, self.max_iter)
        else:
            raise ValueError('You have not selected a valid method for solving the coefficients of the polynomial. Choose from compressed-sensing, least-squares, least-squares-with-gradients, least-absolute-residual, minimum-norm, numerical-integration, huber or elastic-net.')
    def get_solver(self):
        return self.solver

def least_squares(A, b, verbose):
    if np.__version__ < '1.14':
        alpha = np.linalg.lstsq(A, b)
    else:
      alpha = np.linalg.lstsq(A, b, rcond=None)
    if verbose is True:
        print('The condition number of the matrix is '+str(np.linalg.cond(A))+'.')
    return alpha[0], None
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
    return x, None
def orthogonal_linear_system(A, b):
    coefficients = np.dot(A.T, b)
    return coefficients, None
def constrained_least_squares(A, b, C, d, verbose):
    # Size of matrices!
    m, n = A.shape
    p, q = b.shape
    k, l = C.shape
    s, t = d.shape
    # Check that the number of elements in b are equivalent to the number of rows in A
    if m != p:
        raise ValueError( 'solver: error: mismatch in sizes of A and b')
    elif k != s:
        raise ValueError( 'solver: error: mismatch in sizes of C and d')
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
    y1,_ = least_squares(L, d, verbose)
    c = b - np.dot( np.dot(A , Q1) , y1)
    AQ2 = np.dot(A , Q2)
    y2,_ = least_squares(AQ2 , c, verbose)
    x = np.dot(Q1 , y1) + np.dot(Q2 , y2)
    cond = np.linalg.cond(AQ2)
    if verbose is True:
        print('The condition number of the matrix is '+str(cond)+'.')
    return x, None
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
    return np.reshape(x, (len(x),1)), None
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

def least_absolute_residual(A, b, verbose, opt):
    '''
    Solves Ax=b by minimising the L1 norm (absolute residuals).
    '''
    N, d = A.shape
    if verbose: print('Solving for coefficients with least-absolute-residual')

    # Use cvxpy with OSQP for optimising 
    if opt=='osqp':
        if verbose: print('Solving using cvxpy with OSQP solver')
        # Define problem
        b = b.squeeze()
        x = cv.Variable(d)
        objective = cv.sum(cv.abs(A@x - b)) 
        prob = cv.Problem(cv.Minimize(objective))
        # Solve with OSQP
        prob.solve(solver=cv.OSQP,verbose=verbose)
        return x.value, None

    # Use scipy linprog for optimising
    elif opt=='scipy':
        if verbose: print('Solving using scipy linprog')
        c = np.hstack([np.zeros(d), np.ones(N)])
        A1 = np.hstack([A, -np.eye(N)])
        A2 = np.hstack([-A, -np.eye(N)])
        AA = np.vstack([A1, A2])
        bb = np.hstack([b.reshape(-1), -b.reshape(-1)])
        res = linprog(c, A_ub=AA, b_ub=bb)
        return res.x[:d], None

def huber(A, b, verbose, M, opt):
    '''
    Solves Ax=b by minimising the Huber loss function. 
    Thi
    function is identical to the least squares (L2) penalty for small residuals (i.e. ||Ax-b||**2<=M). 
    But on large residuals (||Ax-b||**2>M), its penalty is lower (L1) and increases linearly rather than quadratically. 
    It is thus more forgiving of outliers.
    '''
    if verbose: print('Huber regression with M=%.2f.' %M)

    N,d = A.shape
    if M == None: M = 1.0

    # Use cvxpy with OSQP for optimising
    if opt=='osqp':
        if verbose: print('Solving using cvxpy with OSQP solver')
#       # Define problem as a Quadratic problem
        #       minimize    1/2 z.T * z + M * np.ones(m).T * (r + s)
        #       subject to  Ax - b - z = r - s
        #                   r >= 0
        #                   s >= 0
        # See eq. (24) from https://doi.org/10.1109/34.877518
        b = b.squeeze()
        x = cv.Variable(d)
        z = cv.Variable(N)
        r = cv.Variable(N)
        s = cv.Variable(N)
        objective = cv.Minimize(.5 * cv.sum_squares(z) + M*cv.sum(r + s))
        constraints = [A@x - b - z == r - s,
                       r >= 0, s >= 0]
        prob = cv.Problem(objective, constraints)
        prob.solve(solver=cv.OSQP,verbose=verbose,polish=True)
        x = x.value 

    # Use scipy linprog for optimising
    elif opt=='scipy':
        raise ValueError( 'At present cvxpy, must be installed for huber regression to be selected.')
    return x, None

def elastic_net(A, b, verbose, lamda_val, alpha_val, opt):
    '''
    Solves 0.5*||Ax-b||_2 + lamda*( alpha*||x||_1  + 0.5*(1-alpha)*||x||_2**2).
    Elastic net regression: L2 cost function, with mix of L1 and L2 penalties (scaled by lamda1 and lamda2).
    The penalties shrink the parameter estimates in the hopes of reducing variance, improving prediction accuracy, and aiding interpetation.
    lamda controls amount of penalisation i.e. lamda=0 gives OLS. default is 1. 
    alpha controls L1/L2 penalisation mix. alpha=0 gives ridge regression, alpha=1 gives LASSO. 
    Note, to set lamda1 and lamda 2 directly:
    lamda = lamda1 + lamda2
    alpha = lamda1 / (lamda1 + lamda2)
    '''
    N,d = A.shape
    if lamda_val == None: lamda_val = 1.0
    if alpha_val == None: alpha_val = 0.5 
    if verbose: print('Elastic net regression with lambda=%.2f and alpha=%.2f.' %(lamda_val,alpha_val))

    # Use cvxpy with OSQP for optimising 
    if opt=='osqp':
        if verbose: print('Solving using cvxpy with OSQP solver')
        # Define problem
        b = b.squeeze()
        x = cv.Variable(d)
        lamda1 = cv.Parameter(nonneg=True)
        lamda2 = cv.Parameter(nonneg=True)
        #objective = 0.5*cv.sum_squares(A*x - b) + lamda1*cv.norm1(x) #+ 0.5*lamda2*cv.pnorm(x, p=2)**2
        objective = 0.5*cv.sum_squares(A@x - b) + lamda1*cv.norm1(x) + 0.5*lamda2*cv.sum_squares(x)
        prob = cv.Problem(cv.Minimize(objective))
        # Solve with OSQP
        lamda1.value = lamda_val*alpha_val
        lamda2.value = lamda_val*(1.-alpha_val)
        prob.solve(solver=cv.OSQP,verbose=verbose)
        x = x.value.reshape(-1,1)

    # Use scipy linprog for optimising
    elif opt=='scipy':
        raise ValueError( 'At present cvxpy, must be installed for elastic net regression to be selected.')
    return x, None

def rvm(A, b, max_iter):
    if max_iter is None:
        max_iter = 1000
    K, card = A.shape
    alpha = np.ones(card)
    alpha_0 = 1.0
    remaining_coeff_ind = np.arange(card)
    removed = np.array([], dtype=int)

    Alpha_diag = np.diag(alpha)

    Sigma = np.linalg.inv(alpha_0 * A.T @ A + Alpha_diag)
    mu = alpha_0 * Sigma @ A.T @ b

    Phi = A.copy()

    all_L = []

    for i in range(max_iter):

        C = 1.0 / alpha_0 * np.eye(K) + Phi @ np.diag(1.0 / np.diag(Alpha_diag)) @ Phi.T
        L = -0.5 * (K * np.log(2.0 * np.pi) + np.linalg.slogdet(C)[1] + b.T @ np.linalg.inv(C) @ b)

        all_L.append(L)
        gamma = 1.0 - alpha * np.diag(Sigma)

        remaining_ind = np.where(gamma >= 1e-10)[0]

        removed = np.append(removed, remaining_coeff_ind[np.where(gamma < 1e-10)[0]])
        remaining_coeff_ind = np.setdiff1d(remaining_coeff_ind, remaining_coeff_ind[np.where(gamma < 1e-10)[0]])


        alpha_new = (gamma / mu ** 2)[remaining_ind]
        alpha_0_new = (K - np.sum(gamma)) / np.linalg.norm(b - Phi @ mu) ** 2

        alpha = alpha_new.copy()
        alpha_0 = alpha_0_new
        Phi = Phi[:, remaining_ind]

        Alpha_diag = np.diag(alpha)
        Sigma = np.linalg.inv(alpha_0 * Phi.T @ Phi + Alpha_diag)

        mu = alpha_0 * Sigma @ Phi.T @ b
        if len(all_L) > 1:
            residual = np.abs((all_L[-1] - all_L[-2]) / (all_L[-1] - all_L[0]))
            if residual < 1e-3:
                break

    if i == max_iter - 1: print('WARNING: Maximum iteration limit reached in solver.py')
    mean_coeffs = np.zeros(card)
    mean_coeffs[remaining_coeff_ind] = mu.copy()

    return mean_coeffs, None 

def lasso(Afull, b, verbose, max_iter, IC):
    """
    Performs LASSO using least angle regression. The full regularisation path is computed, with the entire sequence of LARS steps requiring only O(m3 + nm2) computations (the cost of a single least squares regression!), as long as m < n [2]. The set of coefficients with the lowest model selection criteria Cp is then selected according to [3].

    **References**
        1. Efron B., Hastie T., Johnstone I., Tibshirani R., (2004) Least angle regression. The Annals of Statistics, 32(2), 407-499. `Paper <https://projecteuclid.org/download/pdfview_1/euclid.aos/1083178935>`__
        2. Hesterberg, T., Choi, N. H., Meier, L., Fraley, C., (2008) Least angle and l1 penalized regression: A review. Statistics Surveys, 2(0), 61–93. `Paper <https://projecteuclid.org/download/pdfview_1/euclid.ssu/1211317636>`__
        3. Zou, H., Hastie, T., Tibshirani, R., (2007) On the “degrees of freedom” of the lasso. The Annals of Statistics, 35(5), 2173–2192. `Paper <https://projecteuclid.org/download/pdfview_1/euclid.aos/1194461726>`__
    """
#    A = Afull[:,:-1]
    A = Afull
    n,m = A.shape
    b = b.reshape(-1)
    if max_iter is None: 
        max_iter = min(m,n-1)
    else:
        max_iter = min(m,n-1,max_iter)
    if verbose: print('n = %d, m = %d, max_iter = %d' %(n,m,max_iter))

    cur_pred = np.zeros((n,))
    beta = np.zeros((m,))
    sign = np.zeros((m,))
    beta_path = np.zeros((max_iter, m))
    rss = np.zeros(max_iter)

    active_set = set()
    residual = b - cur_pred
    cur_corr = A.transpose().dot(residual)
    j = np.argmax(np.abs(cur_corr), 0)
    active_set.add(j)
    sign[j] = 1
    early_stop = False
    for it in range(max_iter):
        # Residual and its sum of squares
        residual = b - cur_pred
        rss[it] = np.sum(residual * residual)
        pred_from_beta = A.dot(beta)

        # Compute the Gram matrix X'X for the active set (eq. 2.5 in [1])
        X_a = A[:, list(active_set)]
        X_a *= sign[list(active_set)]
        G_a = X_a.transpose().dot(X_a)

        print(np.linalg.cond(G_a))

        # Invert the gram matrix and scale A_a (eq 2.5 in [1] again)
        try: # if inversion fails i.e. G_a singular then early stop
            G_a_inv = np.linalg.inv(G_a)
        except np.linalg.LinAlgError as err:
            if 'Singular matrix' in str(err): 
                early_stop = True
                break
            else:
                raise
        G_a_inv_red_cols = np.sum(G_a_inv, 1)
        A_a = 1 / np.sqrt(np.sum(G_a_inv_red_cols))

        # Compute equianglular vector (eq. 2.6 in [1])
        omega = A_a * G_a_inv_red_cols
        equiangular = X_a.dot(omega)

        # Check G_a has been inverted ok. i.e. G_a^(-1)@G_a = I. If not then stop early.
        test = G_a_inv@G_a 
        off_diag = np.sum(test)-np.trace(test)
        if abs(off_diag) > 1e-7: 
            early_stop = True
            break

        # Current correlation vector and cos_angle vector (eqs. 2.8 and 2.11 in [1])
        cos_angle = A.transpose().dot(equiangular)
        cur_corr = A.transpose().dot(residual)

        # Largest correlation (eq 2.9 in [1])
        largest_abs_correlation = np.abs(cur_corr).max()

        if verbose:
            print('\nIteration %d/%d' %(it+1,max_iter))
            print('RSS', rss[it])
            print('largest_abs_correlation', largest_abs_correlation)

        # Find gamma to use in update (eq. 2.13 in [1])
        gamma = None
        if it < m - 1:
            next_j = None
            next_sign = 0
            for j in range(m):
                if j in active_set:
                    continue
                v0 = (largest_abs_correlation - cur_corr[j]) / (A_a - cos_angle[j]).item()
                v1 = (largest_abs_correlation + cur_corr[j]) / (A_a + cos_angle[j]).item()
                if v0 > 0 and (gamma is None or v0 < gamma):  # This "or" (and below one) makes it LASSO instead of normal LARS, i.e. see eq 3.6 in [1]
                    next_j = j
                    gamma = v0
                    next_sign = 1
                if v1 > 0 and (gamma is None or v1 < gamma):
                    gamma = v1
                    next_j = j
                    next_sign = -1
        # Final iteration (i.e. OLS)
        else:
            gamma = largest_abs_correlation / A_a 

        # Solving sa*sx = sb to get coefficients for current active set. This is probably optimal speed wise. Should be able to do LASSO-LARS in cost O(m3 + nm2) i.e. cost of single lstsq solve (if m<n, see [2]). Maybe look at the scikit-learn approaches involving Cholesky factorisation and updates/downdates? TODO
        sa = X_a
        sb = equiangular * gamma
        if np.__version__ < '1.14':
            sx = np.linalg.lstsq(sa, sb)
        else:
            sx = np.linalg.lstsq(sa, sb, rcond=None)

        # Get coefficients
        for i, j in enumerate(active_set):
            beta[j] += sx[0][i] * sign[j]

        # Calc equivalent regularisation coefficient lamda i.e. lamda*||x||_1. (This isn't strictly nescesary since penalty term is quantified by sum_j(x_j) for LARS LASSO, but useful for comparision with elastic net etc) 

        # Update prediction and active set
        cur_pred = A@beta
        active_set.add(next_j)
        sign[next_j] = next_sign

        beta_path[it, :] = beta
        if verbose:
            print('beta', beta)
            print('max correlation: %s' % (largest_abs_correlation - gamma * A_a))

    # Cut off end of arrays if early stop
    if early_stop:
        if verbose: print('LASSO-LARS stopped early at iteration %d out of %d.' %(it+1,max_iter))
        beta_path = beta_path[:it]
        rss       = rss[:it]
        max_iter  = it

    sum_abs_coeff = np.sum(np.abs(beta_path), 1)

    # Calc information statistic (AIC or BIC)
    df = np.arange(max_iter)+1  #degrees of freedom can be approximated to be the number of LARS steps i.e. the number of non-zero coefficients [3]
#    sigma2 = np.var(b) # sklearn uses this for sigma2 i.e. var(b) instead of var(residuals). Why? TODO 
    sigma2 = np.var(b-cur_pred) 

    if IC=='AIC':
        ic = rss/(n*sigma2) + (2/n)*df
    elif IC=='BIC':
        ic = rss/(n*sigma2) + (np.log(n)/n)*df
   
    # Select the set of coefficients which minimise IC
    if IC is not None:
        idx = np.argmin(ic) 
        beta_opt = beta_path[idx]
        if verbose: print('\nLASSO-LARS found optimum betas with DoF: %d' %(idx+1))
    else:
        beta_opt = beta_path[-1]
        idx = max_iter-1
        ic = np.zeros(max_iter)

    return beta_opt, {'sum_beta':sum_abs_coeff, 'beta':beta_path, 'IC':ic,'opt_idx':idx}
