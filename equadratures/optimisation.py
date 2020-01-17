"""Perform unconstrained or constrained optimisation."""
from equadratures.basis import Basis
from equadratures.poly import Poly
from equadratures.subspaces import Subspaces
from equadratures.parameter import Parameter
from scipy import optimize
from scipy.linalg import null_space
import numpy as np
from scipy.special import comb, factorial
from scipy.stats import linregress
import warnings
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
        elif self.method in ['trust-region', 'omorf']:
            x_opt, f_opt, status = self._trust_region(x0, del_k=kwargs.get('del_k', 1.0), delmin=kwargs.get('delmin', 1.0e-8), \
                        delmax=kwargs.get('delmax', 2.0), eta1=kwargs.get('eta1', 0.1), \
                        eta2=kwargs.get('eta2', 0.7), gam1=kwargs.get('gam1', 0.25), gam2=kwargs.get('gam2', 1.5), \
                        omega_s=kwargs.get('omega_s', 0.414), max_evals=kwargs.get('max_evals', 2000), \
                        d=kwargs.get('d', 2), subspace_method=kwargs.get('subspace_method', 'variable-projection'))
            sol = {'x': x_opt, 'fun': f_opt, 'nfev': self.num_evals, 'status': status}
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
                     sampling_args={'mesh':'user-defined', 'sample-points': S, 'sample-outputs': f})
        self.poly.set_model()
        self.Subs = Subspaces(full_space_poly=self.poly, method='active-subspace', subspace_dimension=self.d)
        if self.subspace_method == 'variable-projection':
            U0 = self.Subs.get_subspace()[:,:self.d]
            self.Subs = Subspaces(method='variable-projection', sample_points=S, sample_outputs=f, \
                      subspace_init=U0, subspace_dimension=self.d, polynomial_degree=2)
            self.W1 = self.Subs.get_subspace()[:, :self.d]
        elif self.subspace_method == 'active-subspaces':
            U0 = self.Subs.get_subspace()[:,1].reshape(-1,1)
            U1 = null_space(U0.T)
            self.W1 = U0
            for i in range(self.d-1):
                R = []
                for j in range(U1.shape[1]):
                    U = np.hstack((self.W1, U1[:, j].reshape(-1,1)))
                    Y = np.dot(S, U)
                    myParameters = [Parameter(distribution='uniform', lower=np.min(Y[:,k]), upper=np.max(Y[:,k]), \
                                  order=2) for k in range(Y.shape[1])]
                    myBasis = Basis('total-order')
                    poly = Poly(myParameters, myBasis, method='least-squares', \
                                sampling_args={'mesh': 'user-defined', 'sample-points':Y, 'sample-outputs':f})
                    poly.set_model()
                    f_eval = poly.get_polyfit(Y)
                    _,_,r,_,_ = linregress(f_eval.flatten(),f.flatten()) 
                    R.append(r**2)
                index = np.argmax(R)
                self.W1 = np.hstack((self.W1, U1[:, index].reshape(-1,1)))
                U1 = np.delete(U1, index, 1)

    def _blackbox_evaluation(self, s):
        """
        Evaluates the point s for ``trust-region`` or ``omorf`` methods
        """
        if s.ndim == 1:
            s = s.reshape(1,-1)
            f = np.array([[self.objective['function'](s.flatten())]])
            self.num_evals += 1
        else:
            f = np.zeros(s.shape[0])
            for i in range(s.shape[0]):
                f[i] = self.objective['function'](s[i,:].flatten())
                self.num_evals += 1
            f = f.reshape(-1,1)
        if self.f.size == 0:
            self.S = s
            self.f = f
        else:
            self.S = np.vstack((self.S, s))
            self.f = np.vstack((self.f, f))
        if s.shape[0] == 1:
            return np.asscalar(f)
        else:
            return f
        
    @staticmethod
    def _remove_point_from_set(S, f, s_old, tol):
        ind_current = np.where(np.linalg.norm(S-s_old, axis=1, ord=np.inf) <= tol)[0]
        S = np.delete(S, ind_current, 0)
        f = np.delete(f, ind_current, 0)
        return S, f
    
    @staticmethod
    def _remove_points_within_TR_from_set(S, f, s_old, del_k):
        ind_within_TR = np.where(np.linalg.norm(S-s_old, axis=1, ord=np.inf) <= del_k)[0]
        S = S[ind_within_TR,:]
        f = f[ind_within_TR]
        return S, f
    
    def _sample_set(self, s_old, f_old, del_k, method='new', S=None, f=None, num=1):
#       Poised constant of algorithm
        if method == 'new':
            S_hat = np.copy(self.S)
            f_hat = np.copy(self.f)
            S_hat, f_hat = self._remove_point_from_set(S_hat, f_hat, s_old, del_k*1.0e-8)
            S_hat, f_hat = self._remove_points_within_TR_from_set(S_hat, f_hat, s_old, del_k)
                
            S = s_old.reshape(1,-1)
            f = np.array([f_old])
            S, f = self._improve_poisedness(S, f, S_hat, f_hat, s_old, del_k, 'new')
        elif method == 'replace':
            if max(np.linalg.norm(S-s_old, axis=1, ord=np.inf)) > self.epsilon*del_k:
                ind_distant = np.argmax(np.linalg.norm(S-s_old, axis=1, ord=np.inf))
                S = np.delete(S, ind_distant, 0)
                f = np.delete(f, ind_distant, 0)
            else:
                S_hat = np.copy(S)
                f_hat = np.copy(f)
                S_hat, fhat = self._remove_point_from_set(S_hat, f_hat, s_old, del_k*1.0e-8)
                
                S = s_old.reshape(1,-1)
                f = np.array([f_old])
                S, f = self._improve_poisedness(S, f, S_hat, f_hat, s_old, del_k, 'improve')
        elif method == 'improve':
            S_hat = np.copy(S)
            f_hat = np.copy(f)
            S_hat, f_hat = self._remove_point_from_set(S_hat, f_hat, s_old, del_k*1.0e-8)
            
            if max(np.linalg.norm(S_hat-s_old, axis=1, ord=np.inf)) > self.epsilon*del_k:
                ind_distant = np.argmax(np.linalg.norm(S_hat-s_old, axis=1, ord=np.inf))
                S_hat = np.delete(S_hat, ind_distant, 0)
                f_hat = np.delete(f_hat, ind_distant, 0)
                
            S = s_old.reshape(1,-1)
            f = np.array([f_old])
            S, f = self._improve_poisedness(S, f, S_hat, f_hat, s_old, del_k, 'improve', num)
        return S, f
    
    def _improve_poisedness(self, S, f, S_hat, f_hat, s_old, del_k, method=None, num=1):
        if self.bounds is not None:
            bounds_l = np.maximum(self.bounds[0], s_old-del_k)
            bounds_u = np.minimum(self.bounds[1], s_old+del_k)
        else:
            bounds_l = s_old-del_k
            bounds_u = s_old+del_k
        xc = 0.5*(bounds_l + bounds_u)
        Del_S = del_k
        if S_hat.shape[0] > 1:
            Del_S = max(np.linalg.norm(S_hat-xc, axis=1, ord=np.inf))
        if self.method == 'trust-region':
            Base = Basis('total-order', orders=np.tile([2], self.n))
            basis = Base.get_basis()[:,range(self.n-1, -1, -1)]
            def phi_function(s):
                s = (s - s_old) / Del_S
                try:
                    m,n = s.shape
                except:
                    m = 1
                    s = s.reshape(1,-1)
                phi = np.zeros((m, basis.shape[0]))
                for i in range(m):
                    for k in range(basis.shape[0]):
                        phi[i,k] = np.prod(np.divide(np.power(s[i,:], basis[k,:]), factorial(basis[k,:])))
                if m == 1:
                    return phi.flatten()
                else:
                    return phi
        elif self.method == 'omorf':
            def phi_function(s):
                s = (s - s_old) / Del_S
                try:
                    m,n = s.shape
                except:
                    m = 1
                    s = s.reshape(1,-1)
                phi = np.zeros((m, self.n+1))
                phi[:, 0] = 1.0
                phi[:, 1:] = s
                if m == 1:
                    return phi.flatten()
                else:
                    return phi
#       Initialise U matrix of LU factorisation of M matrix (see Conn et al.)
        U = np.zeros((self.q,self.q))
        U[0,:] = phi_function(s_old)
#       Perform the LU factorisation algorithm for the rest of the points
        for k in range(1, self.q):
            index = None
#            index2 = None
            v = np.zeros(self.q)
            for j in range(k):
                v[j] = -U[j,k] / U[j,j]
            v[k] = 1.0
#           If there are still points to choose from, find if points meet criterion. If so, use the index to choose 
#           point with given index to be next point in regression/interpolation set
            if f_hat.size > 0:
                M = np.absolute(np.array([np.dot(phi_function(S_hat),v)]).flatten())
                index = np.argmax(M)
#                print(phi_function(Xhat))
                if abs(M[index]) < 1.0e-3:
                    index = None
                elif method =='new':
                    if M[index] < self.psi:
                        index = None
                elif method == 'improve':
                    if f.size == self.q - num:
                        if M[index] < self.psi:
                            index = None
#           If index exists, choose the point with that index and delete it from possible choices
            if index is not None:
                s = S_hat[index,:]
                S = np.vstack((S, s))
                f = np.vstack((f, f_hat[index]))
                S_hat = np.delete(S_hat, index, 0)
                f_hat = np.delete(f_hat, index, 0)
#           If index doesn't exist, solve an optimisation point to find the point in the range which best satisfies criterion
            else:
                s = self._find_new_point(v, phi_function, del_k, s_old)
                S = np.vstack((S, s))
                f = np.vstack((f, self._blackbox_evaluation(s)))
#           Update U factorisation in LU algorithm
            phi = phi_function(s)
            U[k,k] = np.dot(v, phi)
            for i in range(k+1,self.q):
                U[k,i] += phi[i]
                for j in range(k):
                    U[k,i] -= (phi[j]*U[j,i]) / U[j,j]
        return S, f
    
    def _find_new_point(self, v, phi_function, del_k, s_old):
        if self.bounds is not None:
            bounds_l = np.maximum(self.bounds[0], s_old-del_k)
            bounds_u = np.minimum(self.bounds[1], s_old+del_k)
        else:
            bounds_l = s_old-del_k
            bounds_u = s_old+del_k
        bounds = []
        for i in range(self.n):
            bounds.append((bounds_l[i], bounds_u[i]))    
        if self.method == 'trust-region': 
            obj1 = lambda s: np.dot(v, phi_function(s))
            obj2 = lambda s: -np.dot(v, phi_function(s))
            res1 = optimize.minimize(obj1, s_old, method='TNC', \
                                     bounds=bounds, options={'disp': False, 'maxiter': 1000})
            res2 = optimize.minimize(obj2, s_old, method='TNC', \
                                     bounds=bounds, options={'disp': False, 'maxiter': 1000})
            if abs(res1['fun']) > abs(res2['fun']):
                s = res1['x']
            else:
                s = res2['x']
        elif self.method == 'omorf':
            c = v[1:]
            res1 = optimize.linprog(c, bounds=bounds)
            res2 = optimize.linprog(-c, bounds=bounds)
            if abs(np.dot(v, phi_function(res1['x']))) > abs(np.dot(v, phi_function(res2['x']))):
                s = res1['x']
            else:
                s = res2['x']
        return s

    def _build_model(self,S,f):
        """
        Constructs quadratic model for ``trust-region`` or ``omorf`` methods
        """
        if self.method == 'trust-region':
            myParameters = [Parameter(distribution='uniform', lower=np.min(S[:,i]), \
                                      upper=np.max(S[:,i]), order=2) for i in range(self.n)]
            myBasis = Basis('total-order')
            my_poly = Poly(myParameters, myBasis, method='least-squares', \
                           sampling_args={'mesh': 'user-defined', 'sample-points':S, 'sample-outputs':f})
        elif self.method == 'omorf':
            self._calculate_subspace(S, f)
            Y = np.dot(S, self.W1)
            myParameters = [Parameter(distribution='uniform', lower=np.min(Y[:,i]), \
                                      upper=np.max(Y[:,i]), order=2) for i in range(self.d)]
            myBasis = Basis('total-order')
            my_poly = Poly(myParameters, myBasis, method='least-squares', \
                           sampling_args={'mesh': 'user-defined', 'sample-points':Y, 'sample-outputs':f})
        my_poly.set_model()
        return my_poly

    def _compute_step(self,s_old,my_poly,del_k):
        """
        Solves the trust-region subproblem for ``trust-region`` or ``omorf`` methods
        """
        if self.bounds is not None:
            bounds_l = np.maximum(self.bounds[0], s_old-del_k)
            bounds_u = np.minimum(self.bounds[1], s_old+del_k)
        else:
            bounds_l = s_old-del_k
            bounds_u = s_old+del_k
        bounds = []
        for i in range(self.n):
            bounds.append((bounds_l[i], bounds_u[i]))
        if self.method == 'trust-region':
            res = optimize.minimize(lambda x: np.asscalar(my_poly.get_polyfit(x)), s_old, method='TNC', \
                    jac=lambda x: my_poly.get_polyfit_grad(x).flatten(), bounds=bounds, options={'disp': False})
        elif self.method == 'omorf':
            res = optimize.minimize(lambda x: np.asscalar(my_poly.get_polyfit(np.dot(x,self.W1))), s_old, \
                    method='TNC', jac=lambda x: np.dot(self.W1, my_poly.get_polyfit_grad(np.dot(x,self.W1))).flatten(), \
                    bounds=bounds, options={'disp': False})
        s_new = res.x
        m_new = res.fun
        return s_new, m_new
    
    @staticmethod
    def _choose_best(X, f):
        ind_min = np.argmin(f)
        x_0 = X[ind_min,:]
        f_0 = np.asscalar(f[ind_min])
        return x_0, f_0

    def _trust_region(self, s_old, del_k, delmin, delmax, eta1, eta2, gam1, gam2, omega_s, max_evals, \
                      d, subspace_method):
        """
        Computes optimum using either the ``trust-region`` method or the ``omorf`` method
        """
        self.n = s_old.size
        itermax = 10000
        self.epsilon = 1.2
        if self.method == 'trust-region':
            self.psi = 0.25
            self.q = int(comb(self.n+2, 2))
        elif self.method == 'omorf':
            self.d = d
            self.subspace_method = subspace_method
            self.psi = 1.0
            self.epsilon = 1.2
            self.q = self.n + 1
        
        # Make the first black-box function call and initialise the database of solutions and labels
        f_old = self._blackbox_evaluation(s_old)
        # Construct the regression set
        S, f = self._sample_set(s_old, f_old, del_k)
        # Construct the model and evaluate at current point
        s_old, f_old = self._choose_best(self.S, self.f)
        for i in range(itermax):
            # If trust-region radius is less than minimum, break loop
            if len(self.f) >= max_evals or del_k < delmin:
                break
            my_poly = self._build_model(S, f)
            if self.method == 'trust-region':
                m_old = np.asscalar(my_poly.get_polyfit(s_old))
            elif self.method == 'omorf':
                m_old = np.asscalar(my_poly.get_polyfit(np.dot(s_old,self.W1)))
            s_new, m_new = self._compute_step(s_old,my_poly,del_k)
            # Safety step implemented in BOBYQA
            if np.linalg.norm(s_new - s_old, ord=np.inf) < omega_s*del_k:
                del_k *= gam1
                S, f = self._sample_set(s_old, f_old, del_k, 'improve', S, f)
                s_old, f_old = self._choose_best(S, f)
                continue
            f_new = self._blackbox_evaluation(s_new)
            # Calculate trust-region factor
            rho_k = (f_old - f_new) / (m_old - m_new)
            S = np.vstack((S, s_new))
            f = np.vstack((f, f_new))
            s_old, f_old = self._choose_best(S, f)
            if rho_k >= eta2:
                del_k = min(gam2*del_k,delmax)
                S, f = self._sample_set(s_old, f_old, del_k, 'replace', S, f)
            elif rho_k > eta1:
                S, f = self._sample_set(s_old, f_old, del_k, 'replace', S, f)
            else:
                if max(np.linalg.norm(S-s_old, axis=1, ord=np.inf)) <= self.epsilon*del_k:
                    del_k *= gam1
                    S, f = self._sample_set(s_old, f_old, del_k, 'replace', S, f)
                else:
                    S, f = self._sample_set(s_old, f_old, del_k, 'improve', S, f)
        s_old, f_old = self._choose_best(S, f)
        if self.num_evals >= max_evals:
            status = 1
        else:
            status = 0
        return s_old, f_old, status