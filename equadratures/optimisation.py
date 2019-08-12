"""Perform unconstrained or constrained optimisation."""
from scipy import optimize
import numpy as np
class Optimisation:
    """
    This class performs unconstrained or constrained optimisation of poly objects using scipy.optimize.minimize.
    :param string method (optional):
        A string specifying the method that will be used for optimisation. All of the available choices come from scipy.optimize.minimize
        (see <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html> for a list of methods and further information).
        In the case of general constrained optimisation, the options are 'COBYLA', 'SLSQP', and 'trust-constr'. The default is 'trust-constr'.
    """
    def __init__(self, method='trust-constr'):
        self.method = method
        self.objective = {'function': None, 'gradient': None, 'hessian': None}
        self.maximise = False
        self.bounds = None
        self.constraints = []
    def add_objective(self, poly=None, function=None, jac_function=None, hess_function=None, maximise=False):
        """
        Adds objective function to be optimised
        :param poly poly (optional):
            an instance of the poly class
        :param ndarray subspace (optional):
            n-by-d matrix specifying a subspace matrix
        :param function function (optional):
            a lambda function of constraint
        :param function jac_function (optional):
            a lambda function of the gradient of the constraint
        :param function hess_function (optional):
            a lambda function of the Hessian of the constraint
        :param bool maximise (optional):
            a flag to specify if the user would like to maximize
        """
        assert poly is not None or function is not None
        self.maximise = maximise
        if poly is not None:
            k = 1.0
            if self.maximise:
                k = -1.0
            f = poly.get_polyfit_function()
            jac = poly.get_polyfit_grad_function()
            hess = poly.get_polyfit_hess_function()
            function = lambda x: k*np.asscalar(f(x.reshape(1,-1)))
            jac_function = lambda x: k*jac(x.reshape(1,-1))[:,0]
            hess_function = lambda x: k*hess(x.reshape(1,-1))[:,:,0]
        elif function is not None:
            if jac_function is None:
                jac_function = '2-point'
            if hess_function is None:
                hess_function = '2-point'
        self.objective = {'function': function, 'gradient': jac_function, 'hessian': hess_function}
    def add_bounds(self, lb, ub):
        """
        Adds bounds lb <= x <=ub to the optimisation problem.
        Only 'L-BFGS-B', 'TNC', 'SLSQP', 'trust-constr', and 'COBYLA methods can handle bounds.
        :param ndarray lb:
            1-by-n matrix that contains lower bounds of x.
        :param ndarray ub:
            1-by-n matrix that contains upper bounds of x.
        """
        assert lb.size == ub.size
        assert self.method in ['L-BFGS-B', 'TNC', 'SLSQP', 'trust-constr', 'COBYLA']
        if self.method != 'COBYLA':
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
        Adds linear inequality constraints b_l <= A x <= b_u to the optimisation problem.
        Only 'trust-constr', 'COBYLA, and 'SLSQP' methods can handle general constraints.
        :param ndarray A:
            M-by-n matrix that contains coefficients of the linear inequality constraints
        :param ndarray b_l:
            M-by-1 matrix that specifies lower bounds of the linear inequality constraints.
            If there is no lower bound, set b_l = -np.inf * np.ones(M).
        :param ndarray b_u:
            M-by-1 matrix that specifies upper bounds of the linear inequality constraints.
            If there is no upper bound, set b_u = np.inf * np.ones(M).
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
    def add_nonlinear_ineq_con(self, bounds, poly=None, function=None, jac_function=None, hess_function=None):
        """
        Adds nonlinear inequality constraints lb <= g(x) <= ub to the optimisation problem, where lb, ub = bounds.
        Only 'trust-constr', 'COBYLA, and 'SLSQP' methods can handle general constraints.
        If poly object is provided using 'poly' argument, gradients and Hessians will be computed automatically.
        If a lambda function is provided with 'function' argument, the user may provide 'jac_function' for gradients and 'hess_function' for Hessians;
        otherwise, a 2-point differentiation rule will be used to approximate these.
        :param ndarray bounds:
            array with 2 entries specifying the lower and upper bounds of the inequality.
            If there is no lower bound, set bounds[0] = -np.inf.
            If there is no upper bound, set bounds[1] = np.inf.
        :param poly poly (optional):
            an instance of the poly class
        :param function function (optional):
            a lambda function of constraint
        :param function jac_function (optional):
            a lambda function of the gradient of the constraint
        :param function hess_function (optional):
            a lambda function of the Hessian of the constraint
        """
        assert self.method in ['SLSQP', 'trust-constr', 'COBYLA']
        assert poly is not None or Function is not None
        if poly is not None:
            # Get lambda functions for function, gradient, and Hessians from poly object
            g = poly.get_polyfit_function()
            jac = poly.get_polyfit_grad_function()
            hess = poly.get_polyfit_hess_function()
            function = lambda x: g(x.reshape(1,-1))[0]
            jac_function = lambda x: jac(x.reshape(1,-1))[:,0]
            hess_function = lambda x, v: hess(x.reshape(1,-1))[:,:,0]
        elif function is not None:
            if jac_function is None:
                jac_function = '2-point'
            if hess_function is None:
                hess_function = '2-point'
        # trust-constr method has its own nonlinear constraint handler
        if self.method == 'trust-constr':
            self.constraints.append(optimize.NonlinearConstraint(function, bounds[0], bounds[1], jac = jac_function, hess = hess_function))
        # other methods add inequality constraints using dictionary files
        else:
            if not np.isinf(bounds[0]):
                self.constraints.append({'type':'ineq', 'fun': lambda x: function(x) - bounds[0], 'jac': jac_function})
            if not np.isinf(bounds[1]):
                self.constraints.append({'type':'ineq', 'fun': lambda x: -function(x) + bounds[1], 'jac': lambda x: -jac_function(x)})
    def add_linear_eq_con(self, A, b):
        """
        Adds linear equality constraints  A x = b to the optimisation routine.
        Only 'trust-constr' and 'SLSQP' methods can handle equality constraints.
        :param ndarray A:
            M-by-n matrix that contains coefficients of the linear equality constraints
        :param ndarray b:
            M-by-1 matrix that specifies right hand side of the linear equality constraints.
        """
        assert self.method == 'trust-constr' or 'SLSQP'
        if self.method == 'trust-constr':
            self.constraints.append(optimize.LinearConstraint(A,b,b))
        else:
            self.constraints.append({'type':'eq', 'fun': lambda x: A.dot(x) - b, 'jac': lambda x: A})
    def add_nonlinear_eq_con(self, value, poly=None, function=None, jac_function=None, hess_function=None):
        """
        Adds nonlinear inequality constraints g(x) = value to the optimisation routine.
        Only 'trust-constr' and 'SLSQP' methods can handle equality constraints.
        If poly object is provided, gradients and Hessians will be computed automatically.
        If a lambda function is provided with 'function' argument, the user may provide 'jac_function' for gradients and 'hess_function' for Hessians;
        otherwise, a 2-point differentiation rule will be used to approximate these.
        :param float value:
            a float value specifying what nonlinear function must be equal to
        :param poly poly (optional):
            an instance of the poly class
        :param function Function (optional):
            a lambda function of constraint
        :param function jac_function (optional):
            a lambda function of the gradient of the constraint
        :param function hess_function (optional):
            a lambda function of the Hessian of the constraint
        """
        assert self.method == 'trust-constr' or 'SLSQP'
        assert poly is not None or function is not None
        assert value is not None
        if poly is not None:
            g = poly.get_polyfit_function()
            jac = poly.get_polyfit_grad_function()
            hess = poly.get_polyfit_hess_function()
            function = lambda x: np.asscalar(g(x.reshape(1,-1)))
            jac_function = lambda x: jac(x.reshape(1,-1))[:,0]
            hess_function = lambda x, v: hess(x.reshape(1,-1))[:,:,0]
        elif function is not None:
            if jac_function is None:
                jac_function = '2-point'
            if hess_function is None:
                hess_function = '2-point'
        if self.method == 'trust-constr':
            self.constraints.append(optimize.NonlinearConstraint(function, value, value, jac=jac_function, hess=hess_function))
        else:
            self.constraints.append({'type':'eq', 'fun': lambda x: function(x) - value, 'jac': jac_function})
    def optimise_poly(self, x0):
        """
        Performs optimisation on a specified function, provided the objective has been added using 'add_objective' method
        and constraints have been added using the relevant method.
        :param ndarray x0:
            initial point for optimiser
        :return:
            An object containing the optimisation result. Important attributes are: the solution array 'x', a Boolean flag 'success' indicating
            if the optimiser exited successfully, and a doc-string 'message' describing the cause of the termination.
        """
        assert self.objective['function'] is not None
        sol = optimize.minimize(self.objective['function'], x0, method=self.method, bounds = self.bounds, jac=self.objective['gradient'], \
            hess=self.objective['hessian'], constraints=self.constraints, options={'disp': False, 'maxiter': 100000})
        if self.maximise:
            sol['fun'] *= -1.0
        return sol