"""Perform unconstrained or constrained optimization."""
from scipy import optimize
import numpy as np

class Optimization:
    """
    This class performs unconstrained or constrained optimization of Poly objects using scipy.optimize.minimize.

    :param string method (optional):
        A string specifying the method that will be used for optimization. All of the available choices come from scipy.optimize.minimize
        (see <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html> for a list of methods and further information).
        In the case of general constrained optimization, the options are 'COBYLA', 'SLSQP', and 'trust-constr'. The default is 'trust-constr'.
    """
    def __init__(self, method='trust-constr'):
        self.method = method
        self.objective = {'function': None, 'gradient': None, 'hessian': None}
        self.maximize = False
        self.bounds = None
        self.constraints = []

    def addObjective(self, Poly=None, subspace=None, Function=None, jacFunction=None, hessFunction=None, maximize=False):
        """
        :param Poly Poly (optional):
            an instance of the Poly class
        :param ndarray subspace (optional):
            n-by-d matrix specifying a subspace matrix
        :param function Function (optional):
            a lambda function of constraint
        :param function jacFunction (optional):
            a lambda function of the gradient of the constraint
        :param function hessFunction (optional):
            a lambda function of the Hessian of the constraint
        :param bool maximize (optional):
            a flag to specify if the user would like to maximize
        """
        assert Poly is not None or Function is not None
        self.maximize = maximize
        if Poly is not None:
            k = 1.0
            if self.maximize:
                k = -1.0
            f = Poly.getPolyFitFunction()
            jac = Poly.getPolyGradFitFunction()
            hess = Poly.getPolyHessFitFunction()
            if subspace is not None:
                Function = lambda x: k*np.asscalar(f(np.dot(x.reshape(1,-1), subspace)))
                jacFunction = lambda x: k*subspace.dot(jac(x.reshape(1,-1).dot(subspace))[:,0])
                hessFunction = lambda x: k*subspace.dot(hess(x.reshape(1,-1).dot(subspace))[:,:,0]).dot(subspace.T)
            else:
                Function = lambda x: k*np.asscalar(f(x.reshape(1,-1)))
                jacFunction = lambda x: k*jac(x.reshape(1,-1))[:,0]
                hessFunction = lambda x: k*hess(x.reshape(1,-1))[:,:,0]
        elif Function is not None:
            if jacFunction is None:
                jacFunction = '2-point'
            if hessFunction is None:
                hessFunction = '2-point'
        self.objective = {'function': Function, 'gradient': jacFunction, 'hessian': hessFunction}

    def addBounds(self, lb, ub):
        """
        Adds bounds lb <= x <=ub to the optimization problem.

        :param ndarray lb:
            1-by-n matrix that contains lower bounds of x.
        :param ndarray ub:
            1-by-n matrix that contains upper bounds of x.
        """
        assert lb.size == ub.size
        self.bounds = []
        for i in range(lb.size):
            self.bounds.append((lb[i], ub[i]))
        self.bounds = tuple(self.bounds)

    def addLinearIneqCon(self, A, b_l, b_u):
        """
        Adds linear inequality constraints b_l <= A x <= b_u to the optimization problem.
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
#       trust-constr method has its own linear constraint handler
        assert self.method == 'trust-constr' or 'SLSQP' or 'COBYLA'
        if self.method == 'trust-constr':
            self.constraints.append(optimize.LinearConstraint(A,b_l,b_u))
#       other methods add inequality constraints using dictionary files
        else:
            if not np.any(np.isinf(b_l)):
                self.constraints.append({'type':'ineq', 'fun': lambda x: np.dot(A,x) - b_l, 'jac': lambda x: A})
            if not np.any(np.isinf(b_u)):
                self.constraints.append({'type':'ineq', 'fun': lambda x: -np.dot(A,x) + b_u, 'jac': lambda x: -A})

    def addNonLinearIneqCon(self, bounds, Poly=None, subspace=None, Function=None, jacFunction=None, hessFunction=None):
        """
        Adds nonlinear inequality constraints lb <= g(x) <= ub to the optimization problem, where lb, ub = bounds.
        Only 'trust-constr', 'COBYLA, and 'SLSQP' methods can handle general constraints.
        If Poly object is provided using 'poly' argument, gradients and Hessians will be computed automatically.
        If 'subspace' is provided, adds the constraint lb <= g(M^T x) <= ub, where M is a tall matrix specified using the subspace argument.
        If a lambda function is provided with 'function' argument, the user may provide 'jacFunction' for gradients and 'hessFunction' for Hessians;
        otherwise, a 2-point differentiation rule will be used to approximate these.

        :param ndarray bounds:
            array with 2 entries specifying the lower and upper bounds of the inequality.
            If there is no lower bound, set bounds[0] = -np.inf.
            If there is no upper bound, set bounds[1] = np.inf.
        :param Poly Poly (optional):
            an instance of the Poly class
        :param ndarray subspace (optional):
            n-by-d matrix specifying a subspace matrix
        :param function Function (optional):
            a lambda function of constraint
        :param function jacFunction (optional):
            a lambda function of the gradient of the constraint
        :param function hessFunction (optional):
            a lambda function of the Hessian of the constraint
        """
        assert self.method == 'trust-constr' or 'SLSQP' or 'COBYLA'
        assert Poly is not None or Function is not None
        if Poly is not None:
#           Get lambda functions for function, gradient, and Hessians from Poly object
            g = Poly.getPolyFitFunction()
            jac = Poly.getPolyGradFitFunction()
            hess = Poly.getPolyHessFitFunction()
            if subspace is not None:
                Function = lambda x: np.asscalar(g(np.dot(x.flatten(), subspace)))
                jacFunction = lambda x: np.dot(subspace, jac(np.dot(x.flatten(), subspace)).flatten())
                hessFunction = lambda x, v: subspace.dot(hess(np.dot(x.reshape(1,-1), subspace))[:,:,0]).dot(subspace.T)
            else:
                Function = lambda x: g(x.reshape(1,-1))[0]
                jacFunction = lambda x: jac(x.reshape(1,-1))[:,0]
                hessFunction = lambda x, v: hess(x.reshape(1,-1))[:,:,0]
        elif Function is not None:
            if jacFunction is None:
                jacFunction = '2-point'
            if hessFunction is None:
                hessFunction = '2-point'
#       trust-constr method has its own nonlinear constraint handler
        if self.method == 'trust-constr':
            self.constraints.append(optimize.NonlinearConstraint(Function, bounds[0], bounds[1], jac = jacFunction, hess = hessFunction))
#       other methods add inequality constraints using dictionary files
        else:
            if not np.isinf(bounds[0]):
                self.constraints.append({'type':'ineq', 'fun': lambda x: Function(x) - bounds[0], 'jac': jacFunction})
            if not np.isinf(bounds[1]):
                self.constraints.append({'type':'ineq', 'fun': lambda x: -Function(x) + bounds[1], 'jac': lambda x: -jacFunction(x)})

    def addLinearEqCon(self, A, b):
        """
        Adds linear equality constraints  A x = b to the optimization routine.
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

    def addNonLinearEqCon(self, value, Poly=None, subspace=None, Function=None, jacFunction=None, hessFunction=None):
        """
        Adds nonlinear inequality constraints g(x) = value to the optimization routine.
        Only 'trust-constr' and 'SLSQP' methods can handle equality constraints.
        If Poly object is provided, gradients and Hessians will be computed automatically.
        If 'subspace' is provided, this method adds the constraint g(M^T x) = value, where M is a tall matrix specified using the subspace argument.
        If a lambda function is provided with 'function' argument, the user may provide 'jacFunction' for gradients and 'hessFunction' for Hessians;
        otherwise, a 2-point differentiation rule will be used to approximate these.

        :param float value:
            a float value specifying what nonlinear function must be equal to
        :param Poly Poly (optional):
            an instance of the Poly class
        :param ndarray subspace (optional):
            n-by-d matrix specifying the tall subspace matrix
        :param function Function (optional):
            a lambda function of constraint
        :param function jacFunction (optional):
            a lambda function of the gradient of the constraint
        :param function hessFunction (optional):
            a lambda function of the Hessian of the constraint
        """
        assert self.method == 'trust-constr' or 'SLSQP'
        assert Poly is not None or Function is not None
        assert value is not None
        if Poly is not None:
            g = Poly.getPolyFitFunction()
            jac = Poly.getPolyGradFitFunction()
            hess = Poly.getPolyHessFitFunction()
            if subspace is not None:
                Function = lambda x: np.asscalar(g(np.dot(x.reshape(1,-1), subspace)))
                jacFunction = lambda x: subspace.dot(jac(x.reshape(1,-1).dot(subspace))[:,0])
                hessFunction = lambda x, v: subspace.dot(hess(x.reshape(1,-1).dot(subspace))[:,:,0]).dot(subspace.T)
            else:
                Function = lambda x: np.asscalar(g(x.reshape(1,-1)))
                jacFunction = lambda x: jac(x.reshape(1,-1))[:,0]
                hessFunction = lambda x, v: hess(x.reshape(1,-1))[:,:,0]
        elif Function is not None:
            if jacFunction is None:
                jacFunction = '2-point'
            if hessFunction is None:
                hessFunction = '2-point'
        if self.method == 'trust-constr':
            self.constraints.append(optimize.NonlinearConstraint(Function, value, value, jac=jacFunction, hess=hessFunction))
        else:
            self.constraints.append({'type':'eq', 'fun': lambda x: Function(x) - value, 'jac': jacFunction})

    def optimizePoly(self, x0):
        """
        Performs optimization on a specified function, provided the objective has been added using 'addObjective' method
        and constraints have been added using the relevant method.

        :param ndarray x0:
            initial point for optimizer
        :return:
            An object containing the optimization result. Important attributes are: the solution array 'x', a Boolean flag 'success' indicating
            if the optimizer exited successfully, and a doc-string 'message' describing the cause of the termination.
        """
        assert self.objective['function'] is not None
        sol = optimize.minimize(self.objective['function'], x0, method=self.method, bounds = self.bounds, jac=self.objective['gradient'], hess=self.objective['hessian'], constraints=self.constraints, options={'disp': False, 'maxiter': 100000})
        if self.maximize:
            sol['fun'] *= -1.0
        return sol
