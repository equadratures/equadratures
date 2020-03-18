"""Perform unconstrained or constrained optimisation."""
from equadratures.basis import Basis
from equadratures.poly import Poly
from equadratures.subspaces import Subspaces
from equadratures.parameter import Parameter
from scipy import optimize
from scipy.linalg import null_space
import numpy as np
from scipy.special import factorial
from scipy.stats import linregress
import warnings
import sys
import time
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
class Optimisation:
    """
    This class performs unconstrained or constrained optimisation of poly objects or custom functions
    using scipy.optimize.minimize or an in-house trust-region method.
    :param string method: A string specifying the method that will be used for optimisation. All of the available choices come from scipy.optimize.minimize (`click here <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`__ for a list of methods and further information). In the case of general constrained optimisation, the options are ``COBYLA``, ``SLSQP``, and ``trust-constr``. The default is ``trust-constr``.
    """
    def __init__(self, method):
        self.method = method
        self.objective = {'function': None, 'gradient': None, 'hessian': None}
        self.maximise = False
        self.bounds = None
        self.constraints = []
        self.num_evals = 0
        # np.random.seed(42)
        if self.method in ['trust-region', 'omorf']:
            self.num_evals = 0
            self.S = np.array([])
            self.f = np.array([])
            self.g = np.array([])
    def add_objective(self, poly=None, custom=None, maximise=False):
        """
        Adds objective function to be optimised.

        :param poly poly:
            A Poly object.
        :param dict custom: Optional arguments centered around the custom option.
            :callable function: The objective function to be called.
            :callable jac_function: The gradient (or derivative) of the objective.
            :callable hess_function: The Hessian of the objective function.
        :param bool maximise: A flag to specify if the user would like to maximise the function instead of minimising it.
        """
        assert poly is not None or custom is not None
        if self.method == 'trust-region':
            assert poly is None
            assert custom is not None
        self.maximise = maximise
        k = 1.0
        if self.maximise:
            k = -1.0
        if poly is not None:
            f = poly.get_polyfit_function()
            jac = poly.get_polyfit_grad_function()
            hess = poly.get_polyfit_hess_function()
            objective = lambda x: k*np.asscalar(f(x))
            objective_deriv = lambda x: k*jac(x)[:,0]
            objective_hess = lambda x: k*hess(x)[:,:,0]
        elif custom is not None:
            assert 'function' in custom
            objective = lambda s: k*custom['function'](s)
            if 'jac_function' in custom:
                objective_deriv = lambda s: k*custom['jac_function'](s)
            else:
                objective_deriv = '2-point'
            if 'hess_function' in custom:
                objective_hess = lambda s: k*custom['hess_function'](s)
            else:
                objective_hess = optimize.BFGS()
        self.objective = {'function': objective, 'gradient': objective_deriv, 'hessian': objective_hess}
    def add_bounds(self, lb, ub):
        """
        Adds bounds :math:`lb <= x <=ub` to the optimisation problem. Only ``L-BFGS-B``, ``TNC``, ``SLSQP``, ``trust-constr``, ``trust-region``, and ``COBYLA`` methods can handle bounds.

        :param numpy.ndarray lb: 1-by-n matrix that contains lower bounds of x.
        :param numpy.ndarray ub: 1-by-n matrix that contains upper bounds of x.
        """
        assert lb.size == ub.size
        assert self.method in ['L-BFGS-B', 'TNC', 'SLSQP', 'trust-constr', 'COBYLA', 'trust-region', 'omorf']
        if self.method in ['trust-region', 'omorf']:
            self.bounds = [lb, ub]
        elif self.method != 'COBYLA':
            self.bounds = []
            for i in range(lb.size):
                self.bounds.append((lb[i], ub[i]))
            self.bounds = tuple(self.bounds)
        else:
            for factor in range(lb.size):
                if not np.isinf(lb[factor]):
                    l = {'type': 'ineq',
                         'fun': lambda x, i=factor: x[i] - lb[i]}
                    self.constraints.append(l)
                if not np.isinf(ub[factor]):
                    u = {'type': 'ineq',
                         'fun': lambda x, i=factor: ub[i] - x[i]}
                    self.constraints.append(u)
    def add_linear_ineq_con(self, A, b_l, b_u):
        """
        Adds linear inequality constraints :math:`b_l <= A x <= b_u` to the optimisation problem.
        Only ``trust-constr``, ``COBYLA``, and ``SLSQP`` methods can handle general constraints.

        :param numpy.ndarray A: An (M,n) matrix that contains coefficients of the linear inequality constraints.
        :param numpy.ndarray b_l: An (M,1) matrix that specifies lower bounds of the linear inequality constraints. If there is no lower bound, set ``b_l = -np.inf * np.ones(M)``.
        :param numpy.ndarray b_u: A (M,1) matrix that specifies upper bounds of the linear inequality constraints. If there is no upper bound, set ``b_u = np.inf * np.ones(M)``.
        """
        # trust-constr method has its own linear constraint handler
        assert self.method in ['SLSQP', 'trust-constr', 'COBYLA']
        if self.method == 'trust-constr':
            self.constraints.append(optimize.LinearConstraint(A,b_l,b_u))
        # other methods add inequality constraints using dictionary files
        else:
            if not np.any(np.isinf(b_l)):
                self.constraints.append({'type':'ineq', 'fun': lambda x: np.dot(A,x) - b_l, 'jac': lambda x: A})
            if not np.any(np.isinf(b_u)):
                self.constraints.append({'type':'ineq', 'fun': lambda x: -np.dot(A,x) + b_u, 'jac': lambda x: -A})
    def add_nonlinear_ineq_con(self, poly=None, custom=None):
        """
        Adds nonlinear inequality constraints :math:`lb <= g(x) <= ub` (for poly option) with :math:`lb`, :math:`ub = bounds` or :math:`g(x) >= 0` (for function option) to the optimisation problem. Only ``trust-constr``, ``COBYLA``, and ``SLSQP`` methods can handle general constraints.
        If Poly object is provided in the poly dictionary, gradients and Hessians will be computed automatically. If a lambda function is provided in the ``function`` dictionary, the user may also provide ``jac_function`` for gradients and ``hess_function`` for Hessians; otherwise, a 2-point differentiation rule
        will be used to approximate the derivative and a BFGS update will be used to approximate the Hessian.

        :param dict poly: Arguments for poly dictionary.
            :param Poly poly: An instance of the Poly class.
            :param numpy.ndarray bounds: An array with two entries specifying the lower and upper bounds of the inequality. If there is no lower bound, set bounds[0] = -np.inf.If there is no upper bound, set bounds[1] = np.inf.
        :param dict custom: Additional custom callable arguments.
            :callable function: The constraint function to be called.
            :callable jac_function: The gradient (or derivative) of the constraint.
            :callable hess_function: The Hessian of the constraint function.
        """
        assert self.method in ['SLSQP', 'trust-constr', 'COBYLA']
        assert poly is not None or custom is not None
        if poly is not None:
            assert 'bounds' in poly
            bounds = poly['bounds']
            assert 'poly' in poly
            gpoly = poly['poly']
            # Get lambda functions for function, gradient, and Hessians from poly object
            g = gpoly.get_polyfit_function()
            jac = gpoly.get_polyfit_grad_function()
            hess = gpoly.get_polyfit_hess_function()
            constraint = lambda x: g(x)[0]
            constraint_deriv = lambda x: jac(x)[:,0]
            constraint_hess = lambda x, v: hess(x)[:,:,0]
            if self.method == 'trust-constr':
                self.constraints.append(optimize.NonlinearConstraint(constraint, bounds[0], bounds[1], \
                             jac = constraint_deriv, hess = constraint_hess))
            # other methods add inequality constraints using dictionary files
            elif self.method == 'SLSQP':
                if not np.isinf(bounds[0]):
                    self.constraints.append({'type':'ineq', 'fun': lambda x: constraint(x) - bounds[0], \
                             'jac': constraint_deriv})
                if not np.isinf(bounds[1]):
                    self.constraints.append({'type':'ineq', 'fun': lambda x: -constraint(x) + bounds[1], \
                             'jac': lambda x: -constraint_deriv(x)})
            else:
                if not np.isinf(bounds[0]):
                    self.constraints.append({'type':'ineq', 'fun': lambda x: constraint(x) - bounds[0]})
                if not np.isinf(bounds[1]):
                    self.constraints.append({'type':'ineq', 'fun': lambda x: -constraint(x) + bounds[1]})
        elif custom is not None:
            assert 'function' in custom
            constraint = custom['function']
            if 'jac_function' in custom:
                constraint_deriv = custom['jac_function']
            else:
                constraint_deriv = '2-point'
            if 'hess_function' in custom:
                constraint_hess = lambda x, v: custom['hess_function'](x)
            else:
                constraint_hess = optimize.BFGS()
            if self.method == 'trust-constr':
                self.constraints.append(optimize.NonlinearConstraint(constraint, 0.0, np.inf, jac = constraint_deriv, \
                         hess = constraint_hess))
            elif self.method == 'SLSQP':
                if 'jac_function' in custom:
                    self.constraints.append({'type': 'ineq', 'fun': constraint, 'jac': constraint_deriv})
                else:
                    self.constraints.append({'type': 'ineq', 'fun': constraint})
            else:
                self.constraints.append({'type': 'ineq', 'fun': constraint})

    def add_linear_eq_con(self, A, b):
        """
        Adds linear equality constraints  :math:`Ax = b` to the optimisation routine. Only ``trust-constr`` and ``SLSQP`` methods can handle equality constraints.

        :param numpy.ndarray A: A (M, n) matrix that contains coefficients of the linear equality constraints.
        :param numpy.ndarray b: A (M, 1) matrix that specifies right hand side of the linear equality constraints.
        """
        assert self.method == 'trust-constr' or 'SLSQP'
        if self.method == 'trust-constr':
            self.constraints.append(optimize.LinearConstraint(A,b,b))
        else:
            self.constraints.append({'type':'eq', 'fun': lambda x: A.dot(x) - b, 'jac': lambda x: A})
    def add_nonlinear_eq_con(self, poly=None, custom=None):
        """
        Adds nonlinear inequality constraints :math:`g(x) = value` (for poly option) or :math:`g(x) = 0` (for function option) to the optimisation routine.
        Only ``trust-constr`` and ``SLSQP`` methods can handle equality constraints. If poly object is providedin the poly dictionary, gradients and Hessians will be computed automatically.

        :param dict poly: Arguments for poly dictionary.
            :param Poly poly: An instance of the Poly class.
            :param float value: Value of the nonlinear constraint.
        :param dict custom: Additional custom callable arguments.
            :callable function: The constraint function to be called.
            :callable jac_function: The gradient (or derivative) of the constraint.
            :callable hess_function: The Hessian of the constraint function.
        """
        assert self.method == 'trust-constr' or 'SLSQP'
        assert poly is not None or custom is not None
        if poly is not None:
            assert 'value' in poly
            value = poly['value']
            g = poly.get_polyfit_function()
            jac = poly.get_polyfit_grad_function()
            hess = poly.get_polyfit_hess_function()
            constraint = lambda x: np.asscalar(g(x))
            constraint_deriv = lambda x: jac(x)[:,0]
            constraint_hess = lambda x, v: hess(x)[:,:,0]
            if self.method == 'trust-constr':
                self.constraints.append(optimize.NonlinearConstraint(constraint, value, value, jac=constraint_deriv, \
                             hess=constraint_hess))
            else:
                self.constraints.append({'type':'eq', 'fun': lambda x: constraint(x) - value, \
                             'jac': constraint_deriv})
        elif custom is not None:
            assert 'function' in custom
            constraint = custom['function']
            if 'jac_function' in custom:
                constraint_deriv = custom['jac_function']
            else:
                constraint_deriv = '2-point'
            if 'hess_function' in custom:
                constraint_hess = lambda x, v: custom['hess_function'](x)
            else:
                constraint_hess = optimize.BFGS()
            if self.method == 'trust-constr':
                self.constraints.append(optimize.NonlinearConstraint(constraint, 0.0, 0.0, jac=constraint_deriv, \
                         hess=constraint_hess))
            else:
                if 'jac_function' in custom:
                    self.constraints.append({'type':'eq', 'fun': constraint, 'jac': constraint_deriv})
                else:
                    self.constraints.append({'type':'eq', 'fun': constraint})

    def optimise(self, x0, *args, **kwargs):
        """
        Performs optimisation on a specified function, provided the objective has been added using 'add_objective' method
        and constraints have been added using the relevant method.

        :param numpy.ndarray x0: Starting point for optimiser.
        :param float del_k: initial trust-region radius for ``trust-region`` or ``omorf`` methods
        :param float delmin: minimum allowable trust-region radius for ``trust-region`` or ``omorf`` methods
        :param float delmax: maximum allowable trust-region radius for ``trust-region`` or ``omorf`` methods
        :param int d: reduced dimension for ``omorf`` method
        :param string subspace_method: subspace method for ``omorf`` method with options ``variable-projection`` or ``active-subspaces``

        :return:
            **sol**: An object containing the optimisation result. Important attributes are: the solution array ``x``, and a Boolean flag ``success`` indicating
            if the optimiser exited successfully.
        """
        assert self.objective['function'] is not None
        if self.method in ['Newton-CG', 'dogleg', 'trust-ncg', 'trust-krylov', 'trust-exact', 'trust-constr']:
            sol = optimize.minimize(self.objective['function'], x0, method=self.method, bounds = self.bounds, \
                    jac=self.objective['gradient'], hess=self.objective['hessian'], \
                    constraints=self.constraints, options={'disp': False, 'maxiter': 10000})
            sol = {'x': sol['x'], 'fun': sol['fun'], 'nfev': self.num_evals, 'status': sol['status']}
        elif self.method in ['CG', 'BFGS', 'L-BFGS-B', 'TNC', 'SLSQP']:
            sol = optimize.minimize(self.objective['function'], x0, method=self.method, bounds = self.bounds, \
                    jac=self.objective['gradient'], constraints=self.constraints, \
                    options={'disp': False, 'maxiter': 10000})
            sol = {'x': sol['x'], 'fun': sol['fun'], 'nfev': self.num_evals, 'status': sol['status']}
        elif self.method in ['trust-region']:
            self._trust_region(x0, del_k=kwargs.get('del_k', None), del_min=kwargs.get('del_min', 1.0e-8), eta_1=kwargs.get('eta_1', 0.05), eta_2=kwargs.get('eta_2', 0.9), \
                    gam_dec=kwargs.get('gam_dec', 0.5), gam_inc=kwargs.get('gam_inc', 2.0), gam_inc_overline=kwargs.get('gam_inc_overline', 4.0), \
                    alpha_1=kwargs.get('alpha_1', 0.1), alpha_2=kwargs.get('alpha_2', 0.5), omega_s=kwargs.get('omega_s', 0.5),  \
                    max_evals=kwargs.get('max_evals', 10000), random_initial=kwargs.get('random_initial', False), scale_bounds=kwargs.get('scale_bounds', False))
            sol = {'x': self.s_old, 'fun': self.f_old, 'nfev': self.num_evals}
        elif self.method in ['omorf']:
            self._omorf(x0, d=kwargs.get('d', 1), subspace_method=kwargs.get('subspace_method', 'active-subspaces'), \
                    del_k=kwargs.get('del_k', None), del_min=kwargs.get('del_min', 1.0e-8), eta_1=kwargs.get('eta_1', 0.05), eta_2=kwargs.get('eta_2', 0.9), \
                    gam_dec=kwargs.get('gam_dec', 0.98), gam_inc=kwargs.get('gam_inc', 2.0), gam_inc_overline=kwargs.get('gam_inc_overline', 4.0), \
                    alpha_1=kwargs.get('alpha_1', 0.9), alpha_2=kwargs.get('alpha_2', 0.95), omega_s=kwargs.get('omega_s', 0.5), \
                    max_evals=kwargs.get('max_evals', 10000), random_initial=kwargs.get('random_initial', False), scale_bounds=kwargs.get('scale_bounds', False))
            sol = {'x': self.s_old, 'fun': self.f_old, 'nfev': self.num_evals}
        else:
            sol = optimize.minimize(self.objective['function'], x0, method=self.method, bounds = self.bounds, \
                    constraints=self.constraints, options={'disp': False, 'maxiter': 10000})    
            sol = {'x': sol['x'], 'fun': sol['fun'], 'nfev': self.num_evals, 'status': sol['status']}
        if self.maximise:
            sol['fun'] *= -1.0
        return sol
    
    def _calculate_subspace(self, S, f):
        parameters = [Parameter(distribution='uniform', lower=np.min(S[:,i]), upper=np.max(S[:,i]), order=1) for i in range(0, self.n)]
        self.poly = Poly(parameters, basis=Basis('total-order'), method='least-squares', \
                     sampling_args={'sample-points': S, 'sample-outputs': f})
        self.poly.set_model()
        self.Subs = Subspaces(full_space_poly=self.poly, method='active-subspace', subspace_dimension=self.d)
        U0 = self.Subs.get_subspace()[:,0].reshape(-1,1)
        U1 = self.Subs.get_subspace()[:,1:]
        for i in range(self.d-1):
            R = []
            for j in range(U1.shape[1]):
                U = np.hstack((U0, U1[:, j].reshape(-1,1)))
                Y = np.dot(S, U)
                myParameters = [Parameter(distribution='uniform', lower=np.min(Y[:,k]), upper=np.max(Y[:,k]), \
                        order=2) for k in range(Y.shape[1])]
                myBasis = Basis('total-order')
                poly = Poly(myParameters, myBasis, method='least-squares', \
                        sampling_args={'sample-points':Y, 'sample-outputs':f})
                poly.set_model()
                f_eval = poly.get_polyfit(Y)
                _,_,r,_,_ = linregress(f_eval.flatten(),f.flatten()) 
                R.append(r**2)
            index = np.argmax(R)
            U0 = np.hstack((U0, U1[:, index].reshape(-1,1)))
            U1 = np.delete(U1, index, 1)
        if self.subspace_method == 'variable-projection':
            self.Subs = Subspaces(method='variable-projection', sample_points=S, sample_outputs=f, \
                    subspace_init=U0, subspace_dimension=self.d, polynomial_degree=2, tol=1.0e-5, max_iter=2*self.d*self.n)
            self.U = self.Subs.get_subspace()[:, :self.d]
        elif self.subspace_method == 'active-subspaces':
            self.U = U0

    def _blackbox_evaluation(self, s):
        """
        Evaluates the point s for ``trust-region`` or ``omorf`` methods
        """
        s = s.reshape(1,-1)
        f = np.array([[self.objective['function'](self._remove_scaling(s.flatten()))]])
        self.num_evals += 1
        if self.f.size == 0:
            self.S = s
            self.f = f
        else:
            self.S = np.vstack((self.S, s))
            self.f = np.vstack((self.f, f))
        return np.asscalar(f)

    def _generate_initial_set(self):
        """
        Generates an initial set of samples using either coordinate directions or orthogonal, random directions
        """
        if self.random_initial:
            direcs = self._random_directions(self.p, self.bounds_l-self.s_old, self.bounds_u-self.s_old)
        else:
            direcs = self._coordinate_directions(self.p, self.bounds_l-self.s_old, self.bounds_u-self.s_old)
        S = np.zeros((self.p, self.n))
        S[0, :] = self.s_old
        for i in range(1, self.p):
            S[i, :] = self.s_old + np.minimum(np.maximum(self.bounds_l-self.s_old, direcs[i, :]), self.bounds_u-self.s_old)
        return S

    def _coordinate_directions(self, num_pnts, lower, upper):
        """
        Generates coordinate directions
        """
        at_lower_boundary = (lower > -0.01 * self.del_k)
        at_upper_boundary = (upper < 0.01 * self.del_k)
        direcs = np.zeros((num_pnts, self.n))
        for i in range(1, num_pnts):
            if 1 <= i < self.n + 1:
                dirn = i - 1
                step = self.del_k if not at_upper_boundary[dirn] else -self.del_k
                direcs[i, dirn] = step
            elif self.n + 1 <= i < 2*self.n + 1:
                dirn = i - self.n - 1
                step = -self.del_k
                if at_lower_boundary[dirn]:
                    step = min(2.0*self.del_k, upper[dirn])
                if at_upper_boundary[dirn]:
                    step = max(-2.0*self.del_k, lower[dirn])
                direcs[i, dirn] = step
            else:
                itemp = (i - self.n - 1) // self.n
                q = i - itemp*self.n - self.n
                p = q + itemp
                if p > self.n:
                    p, q = q, p - self.n
                direcs[i, p-1] = direcs[p, p-1]
                direcs[i, q-1] = direcs[q, q-1]
        return direcs

    def _random_directions(self, num_pnts, lower, upper):
        """
        Generates orthogonal, random directions
        """
        direcs = np.zeros((self.n, max(2*self.n+1, num_pnts)))
        idx_l = (lower == 0)
        idx_u = (upper == 0)
        active = np.logical_or(idx_l, idx_u)
        inactive = np.logical_not(active)
        nactive = np.sum(active)
        ninactive = self.n - nactive
        if ninactive > 0:
            A = np.random.normal(size=(ninactive, ninactive))
            Qred = np.linalg.qr(A)[0]
            Q = np.zeros((self.n, ninactive))
            Q[inactive, :] = Qred
            for i in range(ninactive):
                scale = self._get_scale(Q[:,i], self.del_k, lower, upper)
                direcs[:, i] = scale * Q[:,i]
                scale = self._get_scale(-Q[:,i], self.del_k, lower, upper)
                direcs[:, self.n+i] = -scale * Q[:,i]
        idx_active = np.where(active)[0]
        for i in range(nactive):
            idx = idx_active[i]
            direcs[idx, ninactive+i] = 1.0 if idx_l[idx] else -1.0
            direcs[:, ninactive+i] = get_scale(direcs[:, ninactive+i], self.del_k, lower, upper) * direcs[:, ninactive+i]
            sign = 1.0 if idx_l[idx] else -1.0
            if upper[idx] - lower[idx] > self.del_k:
                direcs[idx, self.n+ninactive+i] = 2.0*sign*self.del_k
            else:
                direcs[idx, self.n+ninactive+i] = 0.5*sign*(upper[idx] - lower[idx])
            direcs[:, self.n+ninactive+i] = self._get_scale(direcs[:, self.n+ninactive+i], 1.0, lower, upper)*direcs[:, self.n+ninactive+i]
        for i in range(num_pnts - 2*self.n):
            dirn = np.random.normal(size=(self.n,))
            for j in range(nactive):
                idx = idx_active[j]
                sign = 1.0 if idx_l[idx] else -1.0
                if dirn[idx]*sign < 0.0:
                    dirn[idx] *= -1.0
            dirn = dirn / np.linalg.norm(dirn)
            scale = self._get_scale(dirn, self.del_k, lower, upper)
            direcs[:, 2*self.n+i] = dirn * scale
        return np.vstack((np.zeros(self.n), direcs[:, :num_pnts].T))

    @staticmethod
    def _get_scale(dirn, delta, lower, upper):
        scale = delta
        for j in range(len(dirn)):
            if dirn[j] < 0.0:
                scale = min(scale, lower[j] / dirn[j])
            elif dirn[j] > 0.0:
                scale = min(scale, upper[j] / dirn[j])
        return scale
        
    @staticmethod
    def _remove_point_from_set(S, f, s):
        ind_current = np.where(np.linalg.norm(S-s, axis=1, ord=np.inf) == 0.0)[0]
        S = np.delete(S, ind_current, 0)
        f = np.delete(f, ind_current, 0)
        return S, f

    def _choose_closest_points(self, n):
        ind_closest = np.argsort(np.linalg.norm(self.S-self.s_old, axis=1, ord=np.inf), kind='mergesort')[:n]
        S = self.S[ind_closest,:]
        f = self.f[ind_closest]
        return S, f

    def _remove_furthest_point(self, S, f):
        ind_distant = np.argmax(np.linalg.norm(S-self.s_old, axis=1, ord=np.inf))
        S = np.delete(S, ind_distant, 0)
        f = np.delete(f, ind_distant, 0)
        return S, f

    def _remove_points_outside_radius(self, S, f, radius):
        ind_inside = np.where(np.linalg.norm(S-self.s_old, axis=1, ord=np.inf) < radius)[0]
        S = S[ind_inside,:]
        f = f[ind_inside]
        return S, f

    def _update_bounds(self):
        if self.bounds is not None:
            if self.scale_bounds:
                self.bounds_l = np.maximum(np.zeros(self.n), self.s_old-self.del_k)
                self.bounds_u = np.minimum(np.ones(self.n), self.s_old+self.del_k)
            else:
                self.bounds_l = np.maximum(self.bounds[0], self.s_old-self.del_k)
                self.bounds_u = np.minimum(self.bounds[1], self.s_old+self.del_k)
        else:
            self.bounds_l = self.s_old-self.del_k
            self.bounds_u = self.s_old+self.del_k
        return None

    def _apply_scaling(self, S):
        if self.bounds is not None and self.scale_bounds:
            shift = self.bounds[0].copy()
            scale = self.bounds[1] - self.bounds[0]
            return np.divide((S - shift), scale)
        else:
            return S

    def _remove_scaling(self, S):
        if self.bounds is not None and self.scale_bounds:
            shift = self.bounds[0].copy()
            scale = self.bounds[1] - self.bounds[0]
            return shift + np.multiply(S, scale)
        else:
            return S
    
    def _sample_set(self, method, S=None, f=None, s_new=None, f_new=None, full_space=True):
        if full_space:
            q = self.p
        else:
            q = self.q
        if method == 'replace':
            S_hat = np.copy(S) 
            f_hat = np.copy(f)
            S_hat = np.vstack((S_hat, s_new))
            f_hat = np.vstack((f_hat, f_new))
            if S.shape != np.unique(S, axis=0).shape:
                S_hat, index = np.unique(S_hat, axis=0, return_index=True)
                f_hat = f_hat[index]
            elif max(np.linalg.norm(S-self.s_old, axis=1, ord=np.inf)) > max(self.epsilon1*self.del_k, self.epsilon2*self.rho_k):
                S_hat, f_hat = self._remove_furthest_point(S_hat, f_hat)
            S_hat, f_hat = self._remove_point_from_set(S_hat, f_hat, self.s_old)
            S = np.zeros((q, self.n))
            f = np.zeros((q, 1))
            S[0, :] = self.s_old
            f[0, :] = self.f_old
            S, f = self._LU_pivoting(S, f, S_hat, f_hat, full_space)
        elif method == 'improve':
            S_hat = np.copy(S) 
            f_hat = np.copy(f)
            if max(np.linalg.norm(S-self.s_old, axis=1, ord=np.inf)) > max(self.epsilon1*self.del_k, self.epsilon2*self.rho_k):
                S_hat, f_hat = self._remove_furthest_point(S_hat, f_hat)
            S_hat, f_hat = self._remove_point_from_set(S_hat, f_hat, self.s_old)
            S = np.zeros((q, self.n))
            f = np.zeros((q, 1))
            S[0, :] = self.s_old
            f[0, :] = self.f_old
            S, f = self._LU_pivoting(S, f, S_hat, f_hat, full_space, 'improve')
        elif method == 'new':
            S_hat = np.array([])
            f_hat = np.array([])
            S = np.zeros((q, self.n))
            f = np.zeros((q, 1))
            S[0, :] = self.s_old
            f[0, :] = self.f_old
            S, f = self._LU_pivoting(S, f, S_hat, f_hat, full_space)
        elif method == 'best_of_large_set':
            S_hat = np.copy(S) 
            f_hat = np.copy(f)
            S_hat, f_hat = self._remove_point_from_set(S_hat, f_hat, self.s_old)
            S = np.zeros((q, self.n))
            f = np.zeros((q, 1))
            S[0, :] = self.s_old
            f[0, :] = self.f_old
            S, f = self._LU_pivoting(S, f, S_hat, f_hat, full_space)
        elif method == 'best_within_region':
            S_hat, f_hat = self._remove_points_outside_radius(self.S, self.f, max(self.epsilon1*self.del_k, self.epsilon2*self.rho_k))
            S = np.zeros((q, self.n))
            f = np.zeros((q, 1))
            S[0, :] = self.s_old
            f[0, :] = self.f_old
            S, f = self._LU_pivoting(S, f, S_hat, f_hat, full_space)
        return S, f
    
    def _LU_pivoting(self, S, f, S_hat, f_hat, full_space, method=None):
        phi_function, phi_function_deriv = self._get_phi_function_and_derivative(S_hat, full_space)
        if full_space:
            q = self.p
        else:
            q = self.q
#       Initialise U matrix of LU factorisation of M matrix (see Conn et al.)
        U = np.zeros((q,q))
        U[0,:] = phi_function(self.s_old)
#       Perform the LU factorisation algorithm for the rest of the points
        for k in range(1, q):
            flag = True
            v = np.zeros(q)
            for j in range(k):
                v[j] = -U[j,k] / U[j,j]
            v[k] = 1.0
#           If there are still points to choose from, find if points meet criterion. If so, use the index to choose 
#           point with given index to be next point in regression/interpolation set
            if f_hat.size > 0:
                M = np.absolute(np.dot(phi_function(S_hat),v).flatten())
                index = np.argmax(M)
                if (method == 'improve' and k == q - 1) or M[index] < 1.0e-5:
                    flag = False
            else:
                flag = False
#           If index exists, choose the point with that index and delete it from possible choices
            if flag:
                s = S_hat[index,:]
                S[k, :] = s
                f[k, :] = f_hat[index]
                S_hat = np.delete(S_hat, index, 0)
                f_hat = np.delete(f_hat, index, 0)
#           If index doesn't exist, solve an optimisation problem to find the point in the range which best satisfies criterion
            else:
                try:
                    s = self._find_new_point(v, phi_function, phi_function_deriv, full_space)
                except:
                    s = self._find_new_point_alternative(v, phi_function)
                if f_hat.size > 0:
                    if M[index] >= abs(np.dot(v, phi_function(s))):
                        s = S_hat[index,:]
                        S[k, :] = s
                        f[k, :] = f_hat[index]
                        S_hat = np.delete(S_hat, index, 0)
                        f_hat = np.delete(f_hat, index, 0)
                    elif self.S.shape == np.unique(np.vstack((self.S, s)), axis=0).shape:
                        s = self._find_new_point_alternative(v, phi_function)
                        if M[index] >= abs(np.dot(v, phi_function(s))):
                            s = S_hat[index,:]
                            S[k, :] = s
                            f[k, :] = f_hat[index]
                            S_hat = np.delete(S_hat, index, 0)
                            f_hat = np.delete(f_hat, index, 0)
                        else:
                            S[k, :] = s
                            f[k, :] = self._blackbox_evaluation(s)
                    else:
                        S[k, :] = s
                        f[k, :] = self._blackbox_evaluation(s)
                else:
                    if self.S.shape == np.unique(np.vstack((self.S, s)), axis=0).shape:
                        s = self._find_new_point_alternative(v, phi_function)
                        S[k, :] = s
                        f[k, :] = self._blackbox_evaluation(s)
                    else:
                        S[k, :] = s
                        f[k, :] = self._blackbox_evaluation(s)
#           Update U factorisation in LU algorithm
            phi = phi_function(s)
            U[k,k] = np.dot(v, phi)
            for i in range(k+1,q):
                U[k,i] += phi[i]
                for j in range(k):
                    U[k,i] -= (phi[j]*U[j,i]) / U[j,j]
        return S, f

    def _get_phi_function_and_derivative(self, S_hat, full_space):
        Del_S = self.del_k
        if self.method == 'trust-region':
            if S_hat.size > 0:
                Del_S = max(np.linalg.norm(S_hat-self.s_old, axis=1, ord=np.inf))
            def phi_function(s):
                s_tilde = np.divide((s - self.s_old), Del_S)
                try:
                    m,n = s_tilde.shape
                except:
                    m = 1
                    s_tilde = s_tilde.reshape(1,-1)
                phi = np.zeros((m, self.q))
                for k in range(self.q):
                    phi[:,k] = np.prod(np.divide(np.power(s_tilde, self.basis[k,:]), factorial(self.basis[k,:])), axis=1)
                if m == 1:
                    return phi.flatten()
                else:
                    return phi
            def phi_function_deriv(s):
                s_tilde = np.divide((s - self.s_old), Del_S)
                phi_deriv = np.zeros((self.n, self.q))
                for i in range(self.n):
                    for k in range(1, self.q):
                        if self.basis[k, i] != 0.0:
                            tmp = np.zeros(self.n)
                            tmp[i] = 1
                            phi_deriv[i,k] = self.basis[k, i] * np.prod(np.divide(np.power(s_tilde, self.basis[k,:]-tmp), factorial(self.basis[k,:])))
                return np.divide(phi_deriv.T, Del_S).T
        elif self.method == 'omorf' and full_space:
            if S_hat.size > 0:
                Del_S = max(np.linalg.norm(S_hat-self.s_old, axis=1, ord=np.inf))
            def phi_function(s):
                s_tilde = np.divide((s - self.s_old), Del_S)
                try:
                    m,n = s_tilde.shape
                except:
                    m = 1
                    s_tilde = s_tilde.reshape(1,-1)
                phi = np.zeros((m, self.p))
                phi[:, 0] = 1.0
                phi[:, 1:] = s_tilde
                if m == 1:
                    return phi.flatten()
                else:
                    return phi
            phi_function_deriv = None
        elif self.method == 'omorf':
            if S_hat.size > 0:
                Del_S = max(np.linalg.norm(np.dot(S_hat-self.s_old,self.U), axis=1))
            def phi_function(s):
                u = np.divide(np.dot((s - self.s_old), self.U), Del_S)
                try:
                    m,n = u.shape
                except:
                    m = 1
                    u = u.reshape(1,-1)
                phi = np.zeros((m, self.q))
                for k in range(self.q):
                    phi[:,k] = np.prod(np.divide(np.power(u, self.basis[k,:]), factorial(self.basis[k,:])), axis=1)
                if m == 1:
                    return phi.flatten()
                else:
                    return phi
            def phi_function_deriv(s):
                u = np.divide(np.dot((s - self.s_old), self.U), Del_S)
                phi_deriv = np.zeros((self.d, self.q))
                for i in range(self.d):
                    for k in range(1, self.q):
                        if self.basis[k, i] != 0.0:
                            tmp = np.zeros(self.d)
                            tmp[i] = 1
                            phi_deriv[i,k] = self.basis[k, i] * np.prod(np.divide(np.power(u, self.basis[k,:]-tmp), factorial(self.basis[k,:])))
                phi_deriv = np.divide(phi_deriv.T, Del_S).T
                return np.dot(self.U, phi_deriv)
        return phi_function, phi_function_deriv
    
    def _find_new_point(self, v, phi_function, phi_function_deriv, full_space=False):
        bounds = []
        for i in range(self.n):
            bounds.append((self.bounds_l[i], self.bounds_u[i])) 
        if full_space:
            c = v[1:]
            res1 = optimize.linprog(c, bounds=bounds)
            res2 = optimize.linprog(-c, bounds=bounds)
            if abs(np.dot(v, phi_function(res1['x']))) > abs(np.dot(v, phi_function(res2['x']))):
                s = res1['x']
            else:
                s = res2['x']
        else:
            obj1 = lambda s: np.dot(v, phi_function(s))
            jac1 = lambda s: np.dot(phi_function_deriv(s), v)
            obj2 = lambda s: -np.dot(v, phi_function(s))
            jac2 = lambda s: -np.dot(phi_function_deriv(s), v)
            res1 = optimize.minimize(obj1, self.s_old, method='TNC', jac=jac1, \
                    bounds=bounds, options={'disp': False})
            res2 = optimize.minimize(obj2, self.s_old, method='TNC', jac=jac2, \
                    bounds=bounds, options={'disp': False})
            if abs(res1['fun']) > abs(res2['fun']):
                s = res1['x']
            else:
                s = res2['x']
        return s

    def _find_new_point_alternative(self, v, phi_function):
        num = int(0.5*(self.n+1)*(self.n+2))
        direcs = self._coordinate_directions(num, self.bounds_l-self.s_old, self.bounds_u-self.s_old)
        S = np.zeros((num, self.n))
        for i in range(num):
            S[i, :] = self.s_old + np.minimum(np.maximum(self.bounds_l-self.s_old, direcs[i, :]), self.bounds_u-self.s_old)
        M = np.absolute(np.dot(phi_function(S), v).flatten())
        indices = np.argsort(M, kind='mergesort')
        for index in reversed(indices):
            s = S[index,:]
            if self.S.shape != np.unique(np.vstack((self.S, s)), axis=0).shape:
                break
        return s

    def _build_model(self, S, f):
        """
        Constructs quadratic model for ``trust-region`` or ``omorf`` methods
        """
        if self.method == 'trust-region':
            myParameters = [Parameter(distribution='uniform', lower=np.min(S[:,i]), \
                    upper=np.max(S[:,i]), order=2) for i in range(self.n)]
            myBasis = Basis('total-order')
            my_poly = Poly(myParameters, myBasis, method='least-squares', \
                    sampling_args={'sample-points':S, 'sample-outputs':f})
        elif self.method == 'omorf':
            Y = np.dot(S, self.U)
            myParameters = [Parameter(distribution='uniform', lower=np.min(Y[:,i]), \
                    upper=np.max(Y[:,i]), order=2) for i in range(self.d)]
            myBasis = Basis('total-order')
            my_poly = Poly(myParameters, myBasis, method='least-squares', \
                    sampling_args={'sample-points':Y, 'sample-outputs':f})
        my_poly.set_model()
        return my_poly

    def _compute_step(self, my_poly):
        """
        Solves the trust-region subproblem for ``trust-region`` or ``omorf`` methods
        """
        bounds = []
        for i in range(self.n):
            bounds.append((self.bounds_l[i], self.bounds_u[i]))
        if self.method == 'trust-region':
            res = optimize.minimize(lambda x: np.asscalar(my_poly.get_polyfit(x)), self.s_old, method='TNC', \
                    jac=lambda x: my_poly.get_polyfit_grad(x).flatten(), bounds=bounds, options={'disp': False})
        elif self.method == 'omorf':
            res = optimize.minimize(lambda x: np.asscalar(my_poly.get_polyfit(np.dot(x,self.U))), self.s_old, \
                    method='TNC', jac=lambda x: np.dot(self.U, my_poly.get_polyfit_grad(np.dot(x,self.U))).flatten(), \
                    bounds=bounds, options={'disp': False})
        s_new = res.x
        m_new = res.fun
        return s_new, m_new
    
    def _set_iterate(self):
        ind_min = np.argmin(self.f)
        self.s_old = self.S[ind_min,:]
        self.f_old = np.asscalar(self.f[ind_min])
        self._update_bounds()

    def _set_del_k(self, value):
        self.del_k = value
        self._update_bounds()

    def _trust_region(self, s_old, del_k, del_min, eta_1, eta_2, gam_dec, gam_inc, gam_inc_overline, alpha_1, alpha_2, omega_s, max_evals, random_initial, scale_bounds):
        """
        Computes optimum using the ``trust-region`` method
        """
        itermax = 10000
        self.n = s_old.size
        self.q = int(0.5*(self.n+1)*(self.n+2))
        self.p = int(0.5*(self.n+1)*(self.n+2))
        self.random_initial = random_initial
        self.scale_bounds = scale_bounds
        Base = Basis('total-order', orders=np.tile([2], self.n))
        self.basis = Base.get_basis()[:,range(self.n-1, -1, -1)]
        self.epsilon1 = 2.0
        self.epsilon2 = 10.0

        self.s_old = self._apply_scaling(s_old)
        self.f_old = self._blackbox_evaluation(self.s_old)
        if del_k is None:
            if self.bounds is None:
                self._set_del_k(max(0.1*np.linalg.norm(self.s_old, ord=np.inf), 1.0))
            elif self.scale_bounds:
                self._set_del_k(0.1)
            else:
                self._set_del_k(min(0.1*np.linalg.norm(self.bounds[1]-self.bounds[0], ord=np.inf), 1.0))
        else:
            self._set_del_k(del_k)
        self.rho_k = self.del_k
        # Construct the sample set
        S = self._generate_initial_set()
        f = np.zeros((self.p, 1))
        f[0, :] = self.f_old
        for i in range(1, S.shape[0]):
            f[i, :] = self._blackbox_evaluation(S[i, :])
            if self.num_evals >= max_evals:
                self.S = self._remove_scaling(self.S)
                self._set_iterate()
                return
        for i in range(itermax):
            if self.num_evals >= max_evals or self.rho_k < del_min:
                break
            try:
                my_poly = self._build_model(S, f)
            except:
                S, f = self._sample_set('improve', S, f)
                continue
            m_old = np.asscalar(my_poly.get_polyfit(self.s_old))
            s_new, m_new = self._compute_step(my_poly)
            step_dist = np.linalg.norm(s_new - self.s_old, ord=np.inf)
            # Safety step implemented in BOBYQA
            if step_dist < omega_s*self.rho_k:
                self._set_del_k(max(min(gam_dec*self.del_k, step_dist), self.rho_k))
                if max(np.linalg.norm(S-self.s_old, axis=1, ord=np.inf)) > max(self.epsilon1*self.del_k, self.epsilon2*self.rho_k):
                    S, f = self._sample_set('improve', S, f)
                else:
                    if self.del_k == self.rho_k:
                        self._set_del_k(alpha_2*self.rho_k)
                        self.rho_k *= alpha_1
                continue
            elif self.S.shape == np.unique(np.vstack((self.S, s_new)), axis=0).shape:
                ind_repeat = np.argmin(np.linalg.norm(self.S - s_new, ord=np.inf, axis=1))
                f_new = self.f[ind_repeat]
            else:
                f_new = self._blackbox_evaluation(s_new)
            if self.num_evals >= max_evals or self.del_k < del_min:
                break
            S, f = self._sample_set('replace', S, f, s_new, f_new)
            # Calculate trust-region factor
            r_k = (self.f_old - f_new) / (m_old - m_new)
            self._set_iterate()
            if r_k >= eta_2:
                self._set_del_k(max(gam_inc*self.del_k, gam_inc_overline*step_dist))
            elif r_k >= eta_1:
                self._set_del_k(max(gam_dec*self.del_k, step_dist, self.rho_k))
            else:
                self._set_del_k(max(min(gam_dec*self.del_k, step_dist), self.rho_k))
                if max(np.linalg.norm(S-self.s_old, axis=1, ord=np.inf)) > max(self.epsilon1*self.del_k, self.epsilon2*self.rho_k):
                    S, f = self._sample_set('improve', S, f)
                else:
                    if self.del_k == self.rho_k:
                        self._set_del_k(alpha_2*self.rho_k)
                        self.rho_k *= alpha_1
        self.S = self._remove_scaling(self.S)
        self._set_iterate()

    def _omorf(self, s_old, d, subspace_method, del_k, del_min, eta_1, eta_2, gam_dec, gam_inc, gam_inc_overline, alpha_1, alpha_2, omega_s, max_evals, random_initial, scale_bounds):
        """
        Computes optimum using the ``omorf`` method
        """
        itermax = 10000
        self.n = s_old.size
        self.d = d
        self.q = int(0.5*(self.d+1)*(self.d+2))
        self.p = self.n+1
        self.random_initial = random_initial
        self.scale_bounds = scale_bounds
        self.subspace_method = subspace_method
        self.max_evals = max_evals
        Base = Basis('total-order', orders=np.tile([2], self.d))
        self.basis = Base.get_basis()[:,range(self.d-1, -1, -1)]
        self.epsilon1 = 2.0
        self.epsilon2 = 10.0

        self.s_old = self._apply_scaling(s_old)
        self.f_old = self._blackbox_evaluation(self.s_old)
        if del_k is None:
            if self.bounds is None:
                self._set_del_k(max(0.1*np.linalg.norm(self.s_old, ord=np.inf), 1.0))
            elif self.scale_bounds:
                self._set_del_k(0.1)
            else:
                self._set_del_k(min(0.1*np.linalg.norm(self.bounds[1]-self.bounds[0], ord=np.inf), 1.0))
        else:
            self._set_del_k(del_k)
        self.rho_k = self.del_k
        # Construct the sample set
        S_full = self._generate_initial_set()
        f_full = np.zeros((self.p, 1))
        f_full[0, :] = self.f_old
        for i in range(1, S_full.shape[0]):
            f_full[i, :] = self._blackbox_evaluation(S_full[i, :])
            if self.num_evals >= max_evals:
                self.S = self._remove_scaling(self.S)
                self._set_iterate()
                return
        S_full, f_full = self._sample_set('best_within_region', full_space=True)
        self._calculate_subspace(S_full, f_full)
        S_red, f_red= self._sample_set('new', full_space=False)
        flag = True
        for i in range(itermax):
            if self.num_evals >= max_evals or self.rho_k < del_min:
                break
            try:
                my_poly = self._build_model(S_red, f_red)
            except:
                S_red, f_red = self._sample_set('improve', S_red, f_red, full_space=False)
                continue
            m_old = np.asscalar(my_poly.get_polyfit(np.dot(self.s_old,self.U)))
            s_new, m_new = self._compute_step(my_poly)
            step_dist = np.linalg.norm(s_new - self.s_old, ord=np.inf)
            # Safety step implemented in BOBYQA
            if step_dist < omega_s*self.rho_k:
                self._set_del_k(max(min(gam_dec*self.del_k, step_dist), self.rho_k))
                if max(np.linalg.norm(S_red-self.s_old, axis=1, ord=np.inf)) > max(self.epsilon1*self.del_k, self.epsilon2*self.rho_k):
                    S_red, f_red = self._sample_set('improve', S_red, f_red, full_space=False)
                elif max(np.linalg.norm(S_full-self.s_old, axis=1, ord=np.inf)) > max(self.epsilon1*self.del_k, self.epsilon2*self.rho_k):
                    S_full, f_full  = self._sample_set('improve', S_full, f_full)
                    try:
                        self._calculate_subspace(S_full, f_full)
                        flag = True
                    except:
                        pass
                    S_red, f_red  = self._sample_set('best_within_region', full_space=False)
                else:
                    if flag:
                        S_red, f_red  = self._sample_set('new', full_space=False)
                        flag = False
                        continue
                    else:
                        flag = False
                        try:
                            self._calculate_subspace(S_full, f_full)
                            flag = True
                            if self.del_k == self.rho_k:
                                self._set_del_k(alpha_2*self.rho_k)
                                self.rho_k *= alpha_1
                        except:
                            S_full, f_full  = self._sample_set('improve', S_full, f_full)
                            S_red, f_red  = self._sample_set('best_within_region', full_space=False) 
                continue
            if self.S.shape == np.unique(np.vstack((self.S, s_new)), axis=0).shape:
                ind_repeat = np.argmin(np.linalg.norm(self.S - s_new, ord=np.inf, axis=1))
                f_new = self.f[ind_repeat]
            else:
                f_new = self._blackbox_evaluation(s_new)
            if self.num_evals >= max_evals or self.rho_k < del_min:
                break
            S_red, f_red = self._sample_set('replace', S_red, f_red, s_new, f_new, full_space=False)
            S_full, f_full = self._sample_set('replace', S_full, f_full, s_new, f_new)
            # Calculate trust-region factor
            del_f = self.f_old - f_new
            del_m = m_old - m_new
            if abs(del_f) < 100*np.finfo(float).eps and abs(del_m) < 100*np.finfo(float).eps:
                r_k = 1.0
            else:
                r_k = del_f / del_m
            self._set_iterate()
            if r_k >= eta_2:
                self._set_del_k(max(gam_inc*self.del_k, gam_inc_overline*step_dist))
            elif r_k >= eta_1:
                self._set_del_k(max(gam_dec*self.del_k, step_dist, self.rho_k))
            else:
                self._set_del_k(max(min(gam_dec*self.del_k, step_dist), self.rho_k))
                if max(np.linalg.norm(S_red-self.s_old, axis=1, ord=np.inf)) > max(self.epsilon1*self.del_k, self.epsilon2*self.rho_k):
                    S_red, f_red = self._sample_set('improve', S_red, f_red, full_space=False)
                elif max(np.linalg.norm(S_full-self.s_old, axis=1, ord=np.inf)) > max(self.epsilon1*self.del_k, self.epsilon2*self.rho_k):
                    S_full, f_full  = self._sample_set('improve', S_full, f_full)
                    try:
                        self._calculate_subspace(S_full, f_full)
                        flag = True
                    except:
                        pass
                    S_red, f_red  = self._sample_set('best_within_region', full_space=False)
                else:
                    if flag:
                        S_red, f_red  = self._sample_set('new', full_space=False)
                        flag = False
                        continue
                    else:
                        flag = False
                        try:
                            self._calculate_subspace(S_full, f_full)
                            flag = True
                            if self.del_k == self.rho_k:
                                self._set_del_k(alpha_2*self.rho_k)
                                self.rho_k *= alpha_1
                        except:
                            S_full, f_full  = self._sample_set('improve', S_full, f_full)
                            S_red, f_red  = self._sample_set('best_within_region', full_space=False) 
        self.S = self._remove_scaling(self.S)
        self._set_iterate()
        return