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
    :param string method: The method used for solving the linear system. Options include: ``compressed-sensing``, ``least-squares``, ``minimum-norm``, ``numerical-integration``, ``least-squares-with-gradients``, ``least-absolute-residual``, ``huber``, ``elastic-net``, ``elastic-path`` and ``relevance-vector-machine``. 
    :param dict solver_args: Optional arguments centered around the specific solver.
            :param numpy.ndarray noise-level: The noise-level to be used in the basis pursuit de-noising solver.
            :param bool verbose: Default value of this input is set to ``False``; when ``True`` a string is printed to the screen detailing the solver convergence and condition number of the matrix.
    """
    # TODO - update poly solver descriptions
    def __init__(self, method, solver_args):
        self.method = method
        self.solver_args = solver_args
        self.noise_level = None
        self.param1 = None # TODO - rename to something huber specific
        self.verbose = False
        self.max_iter = 100
        self.alpha = 1.0
        self.n_lambdas = 100
        self.lambda_eps = 1e-3
        self.lambda_max = None
        self.tol        = 1e-6
        self.crit = 'AIC'
        self.opt = 'osqp'
        if self.solver_args is not None:
            if 'noise-level' in self.solver_args: self.noise_level = solver_args.get('noise-level')
            if 'param1' in self.solver_args: self.param1 = solver_args.get('param1')
            if 'verbose' in self.solver_args: self.verbose = solver_args.get('verbose')
            if 'max-iter' in self.solver_args: self.max_iter = solver_args.get('max-iter')
            if 'alpha' in self.solver_args: self.alpha = solver_args.get('alpha')
            if 'n-lambdas' in self.solver_args: self.n_lambdas = solver_args.get('n-lambdas')
            if 'lambda-eps' in self.solver_args: self.lambda_eps = solver_args.get('lambda-eps')
            if 'lambda-max' in self.solver_args: self.lambda_max = solver_args.get('lambda-max')
            if 'tol' in self.solver_args: self.tol = solver_args.get('tol')
            if 'select-crit' in self.solver_args: self.crit = solver_args.get('select-crit')
            if 'optimiser' in self.solver_args: self.opt = solver_args.get('optimiser')
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
        elif self.method.lower() == 'elastic-net': #MERGE elastic-net and elastic-path? or get rid of elastic-net? TODO
            self.solver = lambda A, b: elastic_net(A, b, self.verbose, self.lambda_max, self.alpha, self.opt)
        elif self.method.lower() == 'elastic-path':
            self.solver = lambda A, b: elastic_path(A, b, self.verbose, self.max_iter, self.alpha, self.n_lambdas, 
                    self.lambda_eps, self.lambda_max, self.tol, self.crit)
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
    if lamda_val == None: lamda_val = 0.1
    if alpha_val == None: alpha_val = 1.0 
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

def elastic_path(A, b, verbose, max_iter, alpha, n_lamdas, lamda_eps, lamda_max, tol, crit):
    """
    Performs elastic net regression via coordinate descent. The full regularisation path is computed (for a given l1 vs l2 blending parameter alpha), and the set of coefficients with the lowest model selection criteria is then selected according to [2]. 
    Choosing alpha=1 gives LASSO regression, whilst alpha=0 gives ridge regression (however alpha<0.01 is unreliable). 

    **References**
        1. Friedman J., Hastie T., Tibshirani R., (2010) Regularization Paths for Generalized Linear Models via Coordinate Descent. Journal of Statistical Software, 33(1), 1-22. `Paper <https://www.jstatsoft.org/article/view/v033i01>`__
        2. Zou, H., Hastie, T., Tibshirani, R., (2007) On the “degrees of freedom” of the lasso. The Annals of Statistics, 35(5), 2173–2192. `Paper <https://projecteuclid.org/download/pdfview_1/euclid.aos/1194461726>`__
    """
    n,p = A.shape
    b = b.reshape(-1)
    assert alpha >= 0.01, 'elastic-path does not work reliably for alpha<0.01, choose 0>alpha<=1.'

    if crit=='CV':
        nfold = 5
    else:
        nfold = 1

    # Get grid of lambda values to cycle through (in descending order)
    lamdas = _get_lamdas(A,b,n_lamdas,lamda_eps,lamda_max,alpha)

    #Run lasso regression for each lambda (w/ warm start i.e. x is passed back in)
    # Loop through nfolds (nfold=1 unless crit=='CV')
    x_path = np.empty([n_lamdas,p,nfold])
    rss    = np.empty([n_lamdas,nfold])*np.nan
    for fold in range(nfold):
        if verbose: print('Fold %d/%d' %(fold,nfold))
        indices = [int(k) for k in fold*np.ceil(n/5.0) + range(int(np.ceil(n/5.0))) if k<n]
        A_val   = A[indices]
        b_val   = b[indices] 
        A_train = np.delete(A, indices, 0)
        b_train = np.delete(b, indices, 0)
        if len(A_val) == 0:
            continue

        x  = np.zeros(p) # Init coeff vector as zeroes
        for l, lamda in enumerate(lamdas):
            if verbose: print('Running coord. descent for lambda = %.2e (%d/%d)' %(lamda,l,n_lamdas))
            x_path[l,:,fold] = _elastic_net_cd(x,A_train,b_train,lamda,alpha,max_iter,tol,verbose)
    
        # RSS for each lambda. A@x_path.T is the predicted b at each point, for each set of coeffs i.e. dimensions (n_samples,n_lambdas)
        rss[:,fold] = np.sum((A_val@x_path[:,:,fold].T - b_val.reshape(-1,1))**2,axis=0)

    # Calc information statistic (AIC or BIC, or RSS if cross validation)
    if crit!='CV':
        df  = np.count_nonzero(x_path[:,:,fold], axis=1) #degrees of freedom can be approximated to be the number of non-zero coefficients [2]
        # Approx sigma2 using residual from saturated model
        residual = b - A@x_path[-1,:,fold]
        sigma2  = np.var(residual) 
        if crit=='AIC':
            ic = rss[:,fold]/(n*sigma2) + (2/n)*df
        elif crit=='BIC':
            ic = rss[:,fold]/(n*sigma2) + (np.log(n)/n)*df
        ic_std = None

    # "information criterion" is mean rss over nfold's for cross validation approach
    else:
        ic = np.nanmean(rss,axis=1) #nanmean in case last fold had len(A_val) therefore that rss not defined
        ic_std = np.nanstd(rss,axis=1)

    # Select the set of coefficients which minimise IC
    idx = np.argmin(ic) 
    x_best = x_path[idx,:,0]
    if verbose: print('\nUsing %a criterion, optimum LASSO lambda = %.2e' %(IC,lamdas[idx]))

    return x_best, {'lambdas':lamdas, 'x_path':x_path[:,:,0], 'IC':ic,'IC_std':ic_std,'opt_idx':idx}

def _elastic_net_cd(x,A,b,lamda,alpha, max_iter,tol,verbose):
    """
    Private method to perform coordinate descent (with elastic net soft  thresholding) for a given lambda and alpha value.
    Following section 2.6 of [1], the algo does one complete pass over the features, and then for following iterations it only loops over the active set (non-zero coefficients). See commit 2b0af9f58fa5ff1876f76f7aedeaf2a0d7d252c8 for a more simple (but considerably slower for large p) algo. 
    """
    # TODO - covariance updates (see 2.2 of [1]) could provide further speed up...

    # Preliminaries
    b = b.reshape(-1)
    A2 = np.sum(A**2, axis=0)
    dx_tol = tol
    n,p = A.shape

    finish  = False
    success = False
    attempt = 0
    while not success:
        attempt += 1
        if (attempt > 2): 
            print('Non-zero coefficients still changing after two cycles, breaking...')
            break

        for n_iter in range(max_iter):
            x_max = 0.0
            dx_max = 0.0
        
            # Residual
            r = b - A@x

            active_set = set(np.argwhere(x).flatten())
            if n_iter == 0 or finish: #First iter or after convergence, loop through entire set 
                loop_set = set(range(p))
            elif n_iter == 1: # Now only loop through active set (i.e. non-zero coeffs)
                loop_set = active_set

            for j in loop_set:
                r = r + A[:,j]*x[j]
                rho = A[:,j]@r/(A2[j] + lamda*(1-alpha))
                x_prev = x[j]
                if j == 0: # TODO - check p0 is still at index 0 when more than parameter
                    x[j] = rho
                else:
                    x[j] = _soft_threshold(rho,lamda*alpha)
                r = r - A[:,j]*x[j]
                
                # Update changes in coeffs
                if j != 0: # TODO - as above
                    d_x    = abs(x[j] - x_prev)
                    dx_max = max(dx_max,d_x)
                    x_max  = max(x_max,abs(x[j]))
                
            # Convergence check - early stop if converged
            if n_iter == max_iter-1:
                conv_msg = 'Max iterations reached without convergence'
                finish = True
            if x_max == 0.0:  # if all coeff's zero
                conv_msg = 'Convergence after %d iterations, x_max=0' %n_iter
                finish = True
            elif dx_max/x_max < dx_tol: # biggest coord update of this iteration smaller than tolerance
                conv_msg = 'Convergence after %d iterations, d_x: %.2e, tol: %.2e' %(n_iter, dx_max/x_max,dx_tol) 
                finish = True
            # TODO - add further duality gap check from 
            #http://proceedings.mlr.press/v37/fercoq15-supp.pdf
            #l1_reg = lamda * alpha * n # For use w/ duality gap calc.
            #l2_reg = lamda * (1.0 - alpha) * n
            #gap = tol + 1

            # Check final complete cycle doesn't add to active set, if it does complete entire process (this is rare!)
            if finish:
                final_active_set = set(np.argwhere(x).flatten())
                if len(final_active_set-active_set) == 0:
                    if verbose: print(conv_msg)
                    success = True
                else:
                    if verbose: print('Final cycle added non-zero coefficients, restarting coordinate descent')
                break
    return x

def _get_lamdas(A,b,n_lamdas,lamda_eps,lamda_maxmax,alpha):
    eps = np.finfo(np.float64).eps 
    n,p = A.shape
      
    # Get list of lambda's
    Ab = (A.T@b*n).reshape(-1) #*n as sum over n

    # From sec 2.5 (with normalisation factor added)
    Ab /= (np.sum(A**2, axis=0) + eps)
    lamda_max = np.max(np.abs(Ab[1:]))/(n*alpha) #1: in here as not applying regularisation to intercept - TODO - check po at 0 for multiple parameters
    if lamda_maxmax is not None:
        lamda_max = min(lamda_max,lamda_maxmax)
    
    if lamda_max <= np.finfo(float).resolution:
        lamdas = np.empty(n_lamdas)
        lamdas.fill(np.finfo(float).resolution)
        return lamdas
    return np.logspace(np.log10(lamda_max * lamda_eps), np.log10(lamda_max),
                           num=n_lamdas)[::-1]

def _soft_threshold(rho,lamda):
    '''Soft thresholding operator for 1D LASSO in elastic net coordinate descent algoritm'''
    if rho < -lamda:
        return (rho + lamda)
    elif rho > lamda:
        return (rho - lamda)
    else:
        return 0.0
