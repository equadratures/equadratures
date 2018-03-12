#TODO: 2nd derivative?
#TODO: combine methods to detect constraints intelligently

from .parameter import Parameter
from .plotting import contour_plot
import numpy as np
import scipy as sp
import warnings

class Polyopt(object):
    def __init__(self, coeffs, index_set, parameters):
        """
        Class for optimization routines using polynomial expansion
        """
        self.coeffs = coeffs
        self.index_set = index_set
        # Prepare for evaluations, need each variable to have derivative_flag == 1!
        for i in parameters:
            assert i.derivative_flag == 1, "Set derivative flag to 1 to use optimization routines."
        self.polyint = Polyint(parameters, index_set)
        self.parameters = parameters

    def SD(self, max_evals, convergence = None, init_guess = None, maximize = False):
        p = self.polyint
        coeffs = self.coeffs
        # Check that initial guess is present. If none provided, initialize at origin
        if init_guess is None:
            init_guess = np.zeros(len(p.uq_parameters))
        else:
            assert(len(init_guess) == len(p.uq_parameters))
        
        constrained = ["Beta", "Uniform", "Gamma", "Weibull", "TruncatedGaussian", "Exponential"]
        for i in p.uq_parameters:
            if i.param_type in constrained:
                warnings.warn("Attempting uncontrained optimization on constrained variables!")
        
        if not(convergence is None):
            convergence = np.abs(convergence)
        else:
            convergence = 0.005

        x = np.asarray(init_guess, dtype = np.float32)
        self.path = x.copy()
        e, d = eval_func(p, coeffs, x, maximize)
        if (np.linalg.norm(d) < convergence):
            return e, x, 1
        else:
            num_of_evals = 1            
            while (num_of_evals < max_evals):
                search_dir = -1.0 * d / np.linalg.norm(d)
                #TODO: adaptive step size with hessian (actually, how to evaluate that...?)
                
                step_size = line_search(p, search_dir, x, coeffs, maximize)
                
                x += step_size * search_dir
                
                self.path = np.vstack((self.path, x.copy()))
                e, d = eval_func(p, coeffs, x, maximize)
                if (np.linalg.norm(d) < convergence):
                    return e, x, num_of_evals
                num_of_evals += 1
            return e,x,num_of_evals
    
    def SLSQP(self, max_evals, convergence = None, init_guess = None, maximize = False, add_cons = None):
        # Minimize function using SQP
        def func(point):
            if maximize:
                return -eval_func(self.polyint, self.coeffs, point)[0]
            else:
                return eval_func(self.polyint, self.coeffs, point)[0]
        def deriv_func(point): 
            if maximize:
                return -eval_func(self.polyint, self.coeffs, point)[1]
            else:
                return eval_func(self.polyint, self.coeffs, point)[1]
        def make_lambda_1(p, a): return lambda x: np.array([x[p] - a])
        def make_lambda_2(jac): return lambda x: np.array(jac)
        def make_lambda_3(p, b): return lambda x: np.array([b - x[p]])
        def make_lambda_4(jac): return lambda x: np.array(-jac)
        cons = []
        for p in range(len(self.parameters)):
            param = self.parameters[p]
            # If statements... may need improvement!
            a = param.lower
            b = param.upper
            jac = np.zeros(len(self.parameters), dtype = np.float32)
            jac[p] = 1.0
            if param.param_type is "Beta" or param.param_type is "Uniform" or param.param_type is "TruncatedGaussian":
                #Two constraints
                cons.append({'type' : 'ineq',
                         'fun' : make_lambda_1(p,a),
                        'jac': make_lambda_2(jac)})
                cons.append({'type': 'ineq',
                         'fun': make_lambda_3(p, b),
                        'jac': make_lambda_4(jac)})
                
            elif param.param_type is "Exponential" or param.param_type is "Gamma" or param.param_type is "Weibull":
                # One constraint
                cons.append({'type' : 'ineq',
                         'fun' : make_lambda_1(p,a),
                        'jac': make_lambda_2(jac)})
        if not(add_cons is None):
            for i in add_cons:
                cons.append(i)
        if init_guess is None:
            init_guess = [p.lower for p in self.parameters]
        
        self.path = np.asarray(init_guess)
        def track_path(Xk):
            self.path = np.vstack((self.path, Xk.copy()))
        r = sp.optimize.minimize(func, init_guess, method = "SLSQP", jac = deriv_func,
                                 options={'disp': True, 'maxiter': max_evals}, constraints = cons, callback = track_path)
        return r.fun, r.x, r.nit
    
    def cont_plot(self, min_x, max_x, min_y, max_y):
        plot_contour(min_x, max_x, min_y, max_y, self.polyint, self.coeffs, self.path)
            
# 1-D line search with golden sections        
def line_search(polyint, search_dir, current_x, coeffs, maximize):
    # Let step size lies btn 0 and 10
    max_leap = 1.0
    A = 0.0
    D = max_leap
    B = A + 0.382 * (D - A)
    C = D - 0.382 * (D - A)
    no_of_red = 20
    AC = False
    BD = False
    for i in range(no_of_red):
        if AC:
            D = C
            C = B            
            B = A + 0.382 * (D - A)
            f_C = f_B
            f_B,_ = eval_func(polyint, coeffs, current_x + B*search_dir, maximize)
        elif BD:
            A = B
            B = C
            C = D - 0.382 * (D - A)
            f_B = f_C
            f_C,_ = eval_func(polyint, coeffs, current_x + C*search_dir, maximize)
        else:
            B = A + 0.382 * (D - A)
            C = D - 0.382 * (D - A)
            f_B, _ = eval_func(polyint, coeffs, current_x + B*search_dir, maximize)
            f_C, _ = eval_func(polyint, coeffs, current_x + C*search_dir, maximize)

        if f_B <= f_C:
            AC = True
            BD = False
        else:
            BD = True
            AC = False
    
    return (A+D)/2.0 
        
# Evaluates polynomial expansion at point
# point is a list, or a 1-d array
def eval_func(polyint, coeffs, point, negative = False):
    evals, derivs = polyint.getMultivariatePolynomial(point)
    e = np.sum(evals* coeffs, 0)
    d = np.asarray([np.sum(derivs[i]* coeffs, 0) for i in derivs.keys()])
    d = d.T.flatten()
    if negative:
        e = -e
        d = -d
    return e, d

# Plot contour for 2-D function..
# Used for example visualization
def plot_contour(min_x, max_x, min_y, max_y, polyint, coeffs, path):
    assert len(polyint.uq_parameters) == 2, "contour plots only plot 2-D functions!"
    X, Y = np.mgrid[min_x:max_x:50j, min_y:max_y:50j]
    points = np.vstack([X.ravel(), Y.ravel()]).T
    Z = np.reshape(eval_func(polyint, coeffs, points)[0], (50,50))
    contour_plot(X,Y,Z, path_points = path[::1, :])
