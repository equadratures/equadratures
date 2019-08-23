"""Perform unconstrained or constrained optimisation."""
from equadratures.basis import Basis
from equadratures.poly import Poly
from equadratures.parameter import Parameter
from scipy import optimize
import numpy as np
from scipy.special import comb, factorial
import warnings
warnings.filterwarnings('ignore')
class Optimisation:
    """
    This class performs unconstrained or constrained optimisation of poly objects or custom functions
    using scipy.optimize.minimize or an in-house trust-region method.
    :param string method: A string specifying the method that will be used for optimisation. All of the available choices come from scipy.optimize.minimize (`click here <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`__ for a list of methods and further information). In the case of general constrained optimisation, the options are ``COBYLA``, ``SLSQP``, and ``trust-constr``. The default is ``trust-constr``.
    """
    def __init__(self, method='trust-constr'):
        self.method = method
        self.objective = {'function': None, 'gradient': None, 'hessian': None}
        self.maximise = False
        self.bounds = None
        self.constraints = []
        if self.method == 'trust-region':
            self.num_evals = 0
            self.S = np.array([])
            self.f = np.array([])
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
        assert self.method in ['L-BFGS-B', 'TNC', 'SLSQP', 'trust-constr', 'COBYLA', 'trust-region']
        if self.method == 'trust-region':
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
                self.constraints.append(optimize.NonlinearConstraint(constraint, bounds[0], bounds[1], jac = constraint_deriv, hess = constraint_hess))
            # other methods add inequality constraints using dictionary files
            elif self.method == 'SLSQP':
                if not np.isinf(bounds[0]):
                    self.constraints.append({'type':'ineq', 'fun': lambda x: constraint(x) - bounds[0], 'jac': constraint_deriv})
                if not np.isinf(bounds[1]):
                    self.constraints.append({'type':'ineq', 'fun': lambda x: -constraint(x) + bounds[1], 'jac': lambda x: -constraint_deriv(x)})
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
                self.constraints.append(optimize.NonlinearConstraint(constraint, 0.0, np.inf, jac = constraint_deriv, hess = constraint_hess))
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
                self.constraints.append(optimize.NonlinearConstraint(constraint, value, value, jac=constraint_deriv, hess=constraint_hess))
            else:
                self.constraints.append({'type':'eq', 'fun': lambda x: constraint(x) - value, 'jac': constraint_deriv})
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
                self.constraints.append(optimize.NonlinearConstraint(constraint, 0.0, 0.0, jac=constraint_deriv, hess=constraint_hess))
            else:
                if 'jac_function' in custom:
                    self.constraints.append({'type':'eq', 'fun': constraint, 'jac': constraint_deriv})
                else:
                    self.constraints.append({'type':'eq', 'fun': constraint})

    def optimise(self, x0):
        """
        Performs optimisation on a specified function, provided the objective has been added using 'add_objective' method
        and constraints have been added using the relevant method.

        :param numpy.ndarray x0: Starting point for optimiser.

        :return:
            **sol**: An object containing the optimisation result. Important attributes are: the solution array ``x``, a Boolean flag ``success`` indicating
            if the optimiser exited successfully, and a doc-string ``message`` describing the cause of the termination.
        """
        assert self.objective['function'] is not None
        if self.method in ['Newton-CG', 'dogleg', 'trust-ncg', 'trust-krylov', 'trust-exact', 'trust-constr']:
            sol = optimize.minimize(self.objective['function'], x0, method=self.method, bounds = self.bounds, jac=self.objective['gradient'], \
                                    hess=self.objective['hessian'], constraints=self.constraints, options={'disp': False, 'maxiter': 10000})
        elif self.method in ['CG', 'BFGS', 'L-BFGS-B', 'TNC', 'SLSQP']:
            sol = optimize.minimize(self.objective['function'], x0, method=self.method, bounds = self.bounds, jac=self.objective['gradient'], \
                                    constraints=self.constraints, options={'disp': False, 'maxiter': 10000})
        elif self.method == 'trust-region':
            x_opt, f_opt, success = self._trust_region(x0)
            sol = {'x': x_opt, 'fun': f_opt, 'nfev': self.num_evals, 'success': success}
        else:
            sol = optimize.minimize(self.objective['function'], x0, method=self.method, bounds = self.bounds, \
                                    constraints=self.constraints, options={'disp': False, 'maxiter': 10000})
        if self.maximise:
            sol['fun'] *= -1.0
        return sol

    def _blackbox_evaluation(self,s):
        """
        Evaluates the point s for ``trust-region`` method
        """
        f = self.objective['function'](s.flatten())
        self.num_evals += 1
        if self.S.size == 0 and self.f.size == 0:
            self.S = s.reshape(1,-1)
            self.f = np.array([[f]])
        elif self.S.size != 0 and self.f.size != 0:
            self.S = np.vstack((self.S,s.reshape(1,-1)))
            self.f = np.vstack((self.f,np.array([f])))
        else:
            raise ValueError('The arrays of solutions and their corresponding function values are not equivalent!')
        return f

    def _regression_set(self, s_old, f_old, del_k):
        """
        Creates the regression set for ``trust-region`` method
        """
#       Copy database of solutions and remove the current iterate
        S_hat = np.copy(self.S)
        f_hat = np.copy(self.f)
        ind_not_current = np.where(np.linalg.norm(S_hat-s_old,axis=1,ord=np.inf) >= 1.0e-14)[0]
        S_hat = S_hat[ind_not_current,:]
        f_hat = f_hat[ind_not_current]
#       Remove points outside the trust-region
        ind_within_TR = np.where(np.linalg.norm(S_hat-s_old,axis=1,ord=np.inf) <= del_k)[0]
        S_hat = S_hat[ind_within_TR,:]
        f_hat = f_hat[ind_within_TR]
#       If Yhat does not contain at least q points, uniformly generate q points with a d-dimensional hypercube of radius rk around centre
        while S_hat.shape[0] < int(0.7*np.ceil(self.q)):
            s = s_old + np.random.uniform(-del_k, del_k, self.n)
            S_hat = np.vstack((S_hat, s))
            f_hat = np.vstack((f_hat, self._blackbox_evaluation(s)))
#       Centre and scale points
        S_hat -= s_old
        DelS = max(np.linalg.norm(S_hat, axis=1, ord=np.inf))
        S_hat = (1.0/DelS)*S_hat
#       Initialise regression/interpolation points and their corresponding function evaluations
        S = np.zeros(self.n).reshape(1,-1)
        f = np.array([[f_old]])
#       Find well-poised points
        S,f,S_hat,f_hat = self._well_poised_LU(S,f,S_hat,f_hat)
#       Include all of the left-over points
        S = np.vstack((S,S_hat))
        f = np.vstack((f,f_hat))
#       Unscale rand uncentre points
        S = DelS*S +s_old
#       Evaluate newly generated regression/interpolation points which do not have an evaluation value
        for j in range(f.shape[0]):
            if np.isinf(f[j]):
                f[j,0] = self._blackbox_evaluation(S[j,:])
        return S, f

    def _well_poised_LU(self,S,f,S_hat,f_hat):
        """
        Ensures the regression set is well-poised using the LU algorithm (proposed by Andrew Conn) for ``trust-region`` method
        """
#       Poised constant of algorithm
        psi = 1.0
#       Generate natural monomial basis
        Base = Basis('total-order', orders=np.tile([1], self.n))
        basis = Base.get_basis()[:,range(self.n-1, -1, -1)]
        def natural_basis_function(x, basis):
            phi = np.zeros(basis.shape[0])
            for j in range(basis.shape[0]):
                phi[j] = 1.0
                for k in range(basis.shape[1]):
                    phi[j] *= (x[k]**basis[j,k]) / factorial(basis[j,k])
            return phi
        phi_function = lambda x: natural_basis_function(x, basis)
#       Initialise U matrix of LU factorisation of M matrix (see Conn et al.)
        U = np.zeros((self.p,self.p))
#       Initialise the first row of U to the e1 basis vector which corresponds to solution with all zeros
        U[0,0] = 1.0
#       Perform the LU factorisation algorithm for the rest of the points
        for k in range(1,self.p):
            v = np.zeros(self.p)
            for j in range(k):
                v[j] = -U[j,k] / U[j,j]
            v[k] = 1.0
#           If there are still points to choose from, find if points meet criterion. If so, use the index to choose
#           point with given index to be next point in regression/interpolation set
            if S_hat.size != 0:
                M = self._natural_basis_matrix(S_hat,v,phi_function)
                index2 = np.argmax(M)
                if M[index2] < psi:
                    index2 = None
            else:
                index2 = None
#           If index exists, choose the point with that index and delete it from possible choices
            if index2 is not None:
                s = S_hat[index2,:].flatten()
                S = np.vstack((S,s))
                f = np.vstack((f,f_hat[index2].flatten()))
                S_hat = np.delete(S_hat, index2, 0)
                f_hat = np.delete(f_hat, index2, 0)
                phi = phi_function(s.flatten())
#           If index doesn't exist, solve an optimisation point to find the point in the range which best satisfies criterion
            else:
                s = optimize.minimize(lambda x: -abs(np.dot(v,phi_function(x.flatten()))), np.zeros(self.n), method='COBYLA',constraints=[{'type':'ineq', 'fun': lambda x: 1.0 - x},{'type':'ineq', 'fun': lambda x: 1.0 + x}],options={'disp': False})['x'].flatten()
                S = np.vstack((S,s))
                f = np.vstack((f,np.array([np.inf])))
                phi = phi_function(s.flatten())
#           Update U factorisation in LU algorithm
            U[k,k] = np.dot(v,phi)
            for i in range(k+1,self.p):
                U[k,i] += phi[i]
                for j in range(k):
                    U[k,i] -= (phi[j]*U[j,i])/U[j,j]
        return S,f,S_hat,f_hat

    def _natural_basis_matrix(self,S,v,phi):
        """
        Helper function for _well_poised_LU for ``trust-region`` method
        """
        M = []
        for i in range(S.shape[0]):
            M.append(phi(S[i,:].flatten()))
        M = np.array(M)
        Mv_abs = np.absolute(np.dot(M,v))
        return Mv_abs

    def _compute_criticality_measure(self,my_poly,s_old,del_k):
        """
        Computes the criticality measure for ``trust-region`` method
        """
        g_k = my_poly.get_polyfit_grad(s_old).flatten()
        if self.bounds is not None:
            crit = optimize.minimize(lambda x: np.dot(g_k,x-s_old), np.zeros_like(s_old), method='COBYLA',constraints=[{'type':'ineq', 'fun': lambda x: self.bounds[1] - x},{'type':'ineq', 'fun': lambda x: x - self.bounds[0]},{'type':'ineq', 'fun': lambda s: del_k*np.ones_like(s_old) - s + s_old},{'type':'ineq', 'fun': lambda s: del_k*np.ones_like(s_old) + s - s_old}],options={'disp': False})['fun']
            alpha_k = abs(crit) / del_k
        else:
            alpha_k = np.linalg.norm(g_k)
        return alpha_k

    def _build_model(self,S,f,del_k):
        """
        Constructs quadratic model for ``trust-region`` method
        """
        myParameters = [Parameter(distribution='uniform', lower=S[0,i] - del_k, upper=S[0,i] + del_k, order=2) for i in range(S.shape[1])]
        myBasis = Basis('total-order')
        my_poly = Poly(myParameters, myBasis, method='compressive-sensing', sampling_args={'sample-points':S, 'sample-outputs':f})
        my_poly.set_model()
        return my_poly

    def _compute_step(self,s_old,my_poly,del_k):
        """
        Solves the trust-region subproblem for ``trust-region`` method
        """
        Opt = Optimisation(method='SLSQP')
        Opt.add_objective(poly=my_poly)
        if self.bounds is not None:
            Opt.add_bounds(self.bounds[0],self.bounds[1])
        Opt.add_linear_ineq_con(np.eye(s_old.size), s_old-del_k*np.ones(s_old.size), s_old+del_k*np.ones(s_old.size))
        sol = Opt.optimise(s_old)
        s_new = sol['x']
        m_new = sol['fun']
        return s_new, m_new

    def _trust_region(self, s_old, del_k = 1.0, eta0 = 0.0, eta1 = 0.5, gam0 = 0.01, gam1 = 1.5, epsilon_c = 1.0e-2, delkmin = 1.0e-10, delkmax = 2.0):
        """
        Computes optimum using the ``trust-region`` method
        """
        self.n = s_old.size
        self.p = self.n + 1
        self.q = int(comb(self.n+2, 2))
        itermax = 500
#       Make the first black-box function call and initialise the database of solutions and labels
        f_old = self._blackbox_evaluation(s_old)
#       Construct the regression set
        S, f = self._regression_set(s_old,f_old,del_k)
#       Construct the model and evaluate at current point
        my_poly = self._build_model(S,f,del_k)
#       Begin algorithm
        for i in range(itermax):
#           If trust-region radius is less than minimum, break loop
            if del_k < delkmin:
                break
            m_old = np.asscalar(my_poly.get_polyfit(s_old)[0])
#           If gradient of model is very small, need to check the validity of the model
            s_new, m_new = self._compute_step(s_old,my_poly,del_k)
            f_new = self._blackbox_evaluation(s_new)
            if m_new >= m_old:
                del_k *= gam0
                continue
#           Calculate trust-region factor
            rho_k = (f_old - f_new) / (m_old - m_new)
            if rho_k >= eta1:
                s_old = s_new
                f_old = f_new
                S, f = self._regression_set(s_old,f_old,del_k)
                my_poly = self._build_model(S,f,del_k)
                alpha_k = self._compute_criticality_measure(my_poly, s_old, del_k)
                if alpha_k <= epsilon_c:
                    del_k *= gam0
                else:
                    del_k = min(gam1*del_k,delkmax)
            elif rho_k > eta0:
                s_old = s_new
                f_old = f_new
                S, f = self._regression_set(s_old,f_old,del_k)
                my_poly = self._build_model(S,f,del_k)
                alpha_k = self._compute_criticality_measure(my_poly, s_old, del_k)
                if alpha_k <= epsilon_c:
                    del_k *= gam0
            else:
                del_k *= gam0
        alpha_k = self._compute_criticality_measure(my_poly, s_old, del_k)
        if alpha_k < 100.0*epsilon_c:
            success = True
        else:
            success = False
        return s_old, f_old, success