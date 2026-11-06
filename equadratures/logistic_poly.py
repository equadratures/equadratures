from equadratures import *
import numpy as np
from scipy.optimize import minimize
try:
    import pymanopt
    manopt = True
except ImportError as e:
    manopt = False
if manopt:
    from pymanopt.manifolds import Stiefel
    from pymanopt import Problem
    from pymanopt.optimizers import ConjugateGradient

class LogisticPoly(object):
    """
    Class for defining a logistic subspace polynomial, used for classification tasks.

    Parameters
    ----------
    n : optional, int
        Dimension of subspace (should be smaller than ambient input dimension d). Defaults to 2.
    M_init : optional, numpy array of dimensions (d, n)
        Initial guess for subspace matrix. Defaults to a random projection.
    tol : optional, float
        Optimisation terminates when cost function on training data falls below ``tol``. Defaults to 1e-7.
    cauchy_tol : optional, float
        Optimisation terminates when the difference between the average of the last ``cauchy_length`` cost function evaluations and the current cost is below ``cauchy_tol`` times the current evaluation. Defaults to 1e-5.
    cauchy_length : optional, int
        Length of comparison history for Cauchy convergence. Defaults to 3.
    verbosity : optional, one of (0, 1, 2)
        Print debug messages during optimisation. 0 for no messages, 1 for printing final residual every restart, 2 for printing residuals at every iteration. Defaults to 0.
    order : optional, int
        Maximum order of subspace polynomial used. Defaults to 2.
    C : optional, float
        L2 penalty on coefficients. Defaults to 1.0.
    max_M_iters : optional, int
        Maximum optimisation iterations per restart. Defaults to 10.
    restarts : optional, int
        Number of times to restart optimisation. The result with lowest training error is taken at the end. Defaults to 1.


    Examples
    --------
    Fitting and testing a logistic polynomial on a dataset.
        >>> log_quad = eq.LogisticPoly(n=1, cauchy_tol=1e-5, verbosity=0, order=p_order, max_M_iters=100, C=0.001)
        >>> log_quad.fit(X_train, y_train)
        >>> prediction = log_quad.predict(X_test)
        >>> error_rate = np.sum(np.abs(np.round(prediction) - y_test)) / y_test.shape[0]
        >>> print(error_rate)

    """

    def __init__(self, n=2, M_init=None, tol=1e-7, cauchy_tol=1e-5,
                      cauchy_length=3, verbosity=2, order=2, C=1.0, max_M_iters=10, restarts=1):
        if not manopt:
            raise ModuleNotFoundError('pymanopt is required for logistic_poly module.')

        self.n = n
        self.tol = tol
        self.cauchy_tol = cauchy_tol
        self.verbosity = verbosity
        self.cauchy_length = cauchy_length
        self.C = C
        self.max_M_iters = max_M_iters
        self.restarts = restarts
        self.order = order
        self.M_init = M_init
        self.fitted = False

    @staticmethod
    def _sigmoid(U):
        return 1.0 / (1.0 + np.exp(-U))

    def _p(self, X, M, c):
        self.dummy_poly.coefficients = c
        return self.dummy_poly.get_polyfit(X @ M).reshape(-1)

    def _phi(self, X, M, c):
        pW = self._p(X, M, c)
        return self._sigmoid(pW)

    def _cost(self, f, X, M, c):
        this_phi = self._phi(X, M, c)
        return -np.sum(f * np.log(this_phi + 1e-15) + (1.0 - f) * np.log(1 - this_phi + 1e-15)) \
               + 0.5 * self.C * np.linalg.norm(c)**2

    def _dcostdc(self, f, X, M, c):
        W = X @ M
        self.dummy_poly.coefficients = c

        V = self.dummy_poly.get_poly(W)
        # U = self.dummy_poly.get_polyfit(W).reshape(-1)
        diff = f - self._phi(X, M, c)

        return -np.dot(V, diff) + self.C * c

    def _dcostdM(self, f, X, M, c):
        self.dummy_poly.coefficients = c

        W = X @ M
        # U = self.dummy_poly.get_polyfit(W).reshape(-1)
        J = np.array(self.dummy_poly.get_poly_grad(W))
        if len(J.shape) == 2:
            J = J[np.newaxis,:,:]

        diff = f - self._phi(X, M, c)

        Jt = J.transpose((2,0,1))
        XJ = X[:, :, np.newaxis] * np.dot(Jt[:, np.newaxis, :, :], c)

        result = -np.dot(XJ.transpose((1,2,0)), diff)
        return result

    def fit(self, X_train, f_train):
        """ Method to fit logistic polynomial.

        Parameters
        ----------
        X_train : numpy array, shape (N, d)
            Training input points.
        f_train : numpy array, shape (N)
            Training output targets.

        """
        f = f_train
        X = X_train
        tol = self.tol
        d = X_train.shape[1]
        n = self.n

        current_best_residual = np.inf
        my_params = [Parameter(order=self.order, distribution='uniform',
                               lower=-np.sqrt(d), upper=np.sqrt(d)) for _ in range(n)]
        my_basis = Basis('total-order')
        self.dummy_poly = Poly(parameters=my_params, basis=my_basis, method='least-squares')

        for r in range(self.restarts):

            if self.M_init is None:
                M0 = np.linalg.qr(np.random.randn(d, self.n))[0]
            else:
                M0 = self.M_init.copy()

            my_poly_init = Poly(parameters=my_params, basis=my_basis, method='least-squares',
                                sampling_args={'mesh': 'user-defined',
                                               'sample-points': X @ M0,
                                               'sample-outputs': f})
            my_poly_init.set_model()
            c0 = my_poly_init.coefficients.copy()

            residual = self._cost(f, X, M0, c0)

            cauchy_length = self.cauchy_length
            residual_history = []
            iter_ind = 0
            M = M0.copy()
            c = c0.copy()
            while residual > tol:
                if self.verbosity == 2:
                    print('residual = %f' % residual)
                residual_history.append(residual)

                manifold = Stiefel(d, n)
                solver = ConjugateGradient(max_iterations=self.max_M_iters)

                # Minimize over M
                @pymanopt.function.numpy(manifold)
                def func_M(M_var):
                    return self._cost(f, X, M_var, c)
                @pymanopt.function.numpy(manifold)
                def grad_M(M_var):
                    return self._dcostdM(f, X, M_var, c)

                problem = Problem(manifold=manifold, cost=func_M, euclidean_gradient=grad_M)

                M = solver.run(problem, initial_point=M).point

                # Minimize over c
                func_c = lambda c_var: self._cost(f, X, M, c_var)
                grad_c = lambda c_var: self._dcostdc(f, X, M, c_var)

                res = minimize(func_c, x0=c, method='CG', jac=grad_c)
                c = res.x
                residual = self._cost(f, X, M, c)
                if iter_ind < cauchy_length:
                    iter_ind += 1
                elif np.abs(np.mean(residual_history[-cauchy_length:]) - residual)/residual < self.cauchy_tol:
                    break

            if self.verbosity > 0:
                print('Final residual on training data: %f' % self._cost(f, X, M, c))
            if residual < current_best_residual:
                self.M = M
                self.c = c
                current_best_residual = residual
        self.fitted = True

    def predict(self, X):
        """ Method to predict from input test points.

        Parameters
        ----------
        X : numpy array, shape (N, d)
            Test input points.

        Returns
        ----------
        numpy array, shape (N)
            Predictions at specified test points.

        """
        if not self.fitted:
            raise ValueError('Call fit() to fit logistic polynomial first.')
        return self._phi(X, self.M, self.c)


