"""Perform unconstrained or constrained optimisation."""
from scipy import optimize
import numpy as np
class Optimisation:
    """
    This class performs unconstrained or constrained optimisation of poly objects using scipy.optimize.minimize.

    :param string method:
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
    def add_objective(self, poly=None, custom=None, maximise=False):
        """
        Adds objective function to be optimised
        :param poly poly (optional):
            an instance of the poly class
        :param dict custom (optional):
            Arguments for custom option.
            :function function: lambda function for objective function
            :function jac_function (optional): lambda function for derivative of the objective function
            :function hess_function (optional): lambda function for Hessian of the objective function
            a dictionary containing
        :param bool maximise (optional):
            a flag to specify if the user would like to maximize
        """
        assert poly is not None or custom is not None
        self.maximise = maximise
        if poly is not None:
            k = 1.0
            if self.maximise:
                k = -1.0
            f = poly.get_polyfit_function()
            jac = poly.get_polyfit_grad_function()
            hess = poly.get_polyfit_hess_function()
            objective = lambda x: k*np.asscalar(f(x))
            objective_deriv = lambda x: k*jac(x)[:,0]
            objective_hess = lambda x: k*hess(x)[:,:,0]
        elif custom is not None:
            assert 'function' in custom
            objective = custom['function']
            if 'jac_function' in custom:
                objective_deriv = custom['jac_function']
            else:
                objective_deriv = False
            if 'hess_function' in custom:
                objective_hess = custom['hess_function']
            else:
                objective_hess = False
        self.objective = {'function': objective, 'gradient': objective_deriv, 'hessian': objective_hess}
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
    def add_nonlinear_ineq_con(self, poly=None, custom=None):
        """
        Adds nonlinear inequality constraints lb <= g(x) <= ub (for poly option) with lb, ub = bounds or
        g(x) >= 0 (for function option) to the optimisation problem.
        Only 'trust-constr', 'COBYLA, and 'SLSQP' methods can handle general constraints.
        If Poly object is provided in the poly dictionary, gradients and Hessians will be computed automatically.
        If a lambda function is provided in the 'function' dictionary, the user may also provide
        'jac_function' for gradients and 'hess_function' for Hessians; otherwise, a 2-point differentiation rule
        will be used to approximate the derivative and a BFGS update will be used to approximate the Hessian.
        :param poly dictionary (optional):
            Arguments for poly dictionary
            :poly poly: an instance of the poly class
            :ndarray bounds: array with 2 entries specifying the lower and upper bounds of the inequality.
            If there is no lower bound, set bounds[0] = -np.inf.If there is no upper bound, set bounds[1] = np.inf.
        :param dict custom (optional):
            Arguments for custom dictionary
            :function function: a lambda function of the constraint
            :function jac_function (optional): a lambda function of the gradient of the constraint
            :function hess_function (optional): a lambda function of the Hessian of the constraint
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
    def add_nonlinear_eq_con(self, poly=None, custom=None):
        """
        Adds nonlinear inequality constraints g(x) = value (for poly option) or
        g(x) = 0 (for function option) to the optimisation routine.
        Only 'trust-constr' and 'SLSQP' methods can handle equality constraints.
        If poly object is providedin the poly dictionary, gradients and Hessians will be computed automatically.
        If a lambda function is provided in the function dictionary, the user may provide
        'jac_function' for gradients and 'hess_function' for Hessians;
        otherwise, a 2-point differentiation rule will be used to approximate the gradient
        and a BFGS update strategy will be used for the Hessian.
        :param poly dictionary (optional):
            Arguments for poly dictionary
            :poly poly: an instance of the poly class
            :float value: a float value specifying what nonlinear function must be equal to
        :param custom dictionary (optional):
            Arguments for function dictionary
            :function function: a lambda function of constraint
            :function jac_function (optional): a lambda function of the gradient of the constraint
            :function hess_function (optional): a lambda function of the Hessian of the constraint
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
                constraint_hess = custom['hess_function']
            else:
                constraint_hess = optimize.BFGS()
            if self.method == 'trust-constr':
                self.constraints.append(optimize.NonlinearConstraint(constraint, 0.0, 0.0, jac=constraint_deriv, hess=constraint_hess))
            else:
                if 'jac_function' in custom:
                    self.constraint.append({'type':'eq', 'fun': constraint, 'jac': constraint_deriv})
                else:
                    self.constraint.append({'type':'eq', 'fun': constraint})

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
        if self.method in ['Newton-CG', 'dogleg', 'trust-ncg', 'trust-krylov', 'trust-exact', 'trust-constr']:
            sol = optimize.minimize(self.objective['function'], x0, method=self.method, bounds = self.bounds, jac=self.objective['gradient'], \
                                    hess=self.objective['hessian'], constraints=self.constraints, options={'disp': False, 'maxiter': 10000})
        elif self.method in ['CG', 'BFGS', 'L-BFGS-B', 'TNC', 'SLSQP']:
            sol = optimize.minimize(self.objective['function'], x0, method=self.method, bounds = self.bounds, jac=self.objective['gradient'], \
                                    constraints=self.constraints, options={'disp': False, 'maxiter': 10000})
        else:
            sol = optimize.minimize(self.objective['function'], x0, method=self.method, bounds = self.bounds, \
                                    constraints=self.constraints, options={'disp': False, 'maxiter': 10000})
        if self.maximise:
            sol['fun'] *= -1.0
        return sol