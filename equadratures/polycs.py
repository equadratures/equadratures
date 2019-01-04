"""Finding coefficients via compressive sensing."""
from .parameter import Parameter
from .basis import Basis
from .poly import Poly
import numpy as np
from .stats import Statistics, getAllSobol
from .convex import *
import scipy
from .utils import columnNormalize


class Polycs(Poly):
    """
    This class defines a Polycs (polynomial via compressive sensing) object.

    :param array training_inputs: Points at which the model is evaluated.
    :param IndexSet basis: An instance of the IndexSet class, in case the user wants to overwrite the indices that are obtained using the orders of the univariate parameters in Parameters uq_parameters. The latter corresponds to a tensor grid index set and is the default option if no basis parameter input is given.
    :param list parameters: List of instances of Parameters class.
    :param array training_outputs: Column vector of regression targets corresponding to each row of training_inputs. Either this or fun should be specified, but not both. If this is specified, training_inputs must also be specified.
    :param callable fun: Function to evaluate training_inputs on to obtain regression targets automatically. Either this or fun should be specified, but not both.
    :param str sampling: Sampling method for non-data-driven model. Valid sampling methods include "standard", "asymptotic" and "dlm".
    :param str quadrature_rule: Quadrature rule. Can be 'qmc' or 'tensor grid'. Refer to documentation for poly.py for more details.
    :param int no_of_quad_points: Number of quadrature points in case qmc is chosen for quadrature rule. Default number is 10000. Recommended to set to a low number (e.g. 1) when quadrature points are not needed (when statistics need not be computed, for instance).
    
    """
    # Constructor
    def __init__(self, parameters, basis, training_inputs=None, sampling=None, no_of_points=None, fun=None, training_outputs=None, quadrature_rule = None, no_of_quad_points = None):
        super(Polycs, self).__init__(parameters, basis)
        if not(training_inputs is None):
            self.x = training_inputs
            assert self.x.shape[1] == len(parameters) # Check that x is in the correct shape
            self.w = np.eye(self.x.shape[0])
        else:
            if not(training_outputs is None):
                raise ValueError("Must specify training_inputs for data-driven model.")
            self.x, self.w = samplingMethod(self.parameters, self.basis, sampling, no_of_points)
        if not((training_outputs is None) ^ (fun is None)):
            raise ValueError("Specify only one of fun or training_y.")
        if not(fun is None):
            try:
                self.y = np.dot(self.w, np.apply_along_axis(fun, 1, self.x))
            except:
                raise ValueError("Fun must be callable.")
        else:
            self.y = np.dot(self.w, training_outputs)
        if self.dimensions != self.basis.elements.shape[1]:
            raise(ValueError, 'Polycs:__init__:: The number of parameters and the number of dimensions in the index set must be the same.')
        self.setDesignMatrix()
        self.cond = np.linalg.cond(self.A)
        self.coherence = coherence(self.A)
        self.y = np.reshape(self.y, (len(self.y), 1)) 
        self.computeCoefficients()
        self.quadrature_rule = quadrature_rule
        self.getQuadraturePointsWeights(no_of_quad_points)

    def setDesignMatrix(self):
        """
        Sets the design matrix using the polynomials defined in the basis.

        :param Polycs self:
            An instance of the Polycs class.
        """
        self.A = self.getPolynomial(self.x).T
        self.A = np.dot(self.w, self.A)
        super(Polycs, self).__setDesignMatrix__(self.A)

    def computeCoefficients(self):
        """
        This function computes the coefficients using CS. To access the coefficients simply use the class's attribute self.coefficients.

        :param Polycs self:
            An instance of the Polycs class.
        """
        A = self.A
        y = self.y.flatten()
        N = A.shape[0]

        # Possible noise levels
        log_epsilon = [-8,-7,-6,-5,-4,-3,-2,-1]
        epsilon = [float(10**i) for i in log_epsilon]
        errors = np.zeros(5)
        mean_errors = np.zeros(len(epsilon))
        
        # 5 fold cross validation
        for e in range(len(epsilon)):
            for n in range(5):
                indices = [int(i) for i in n * np.ceil(N/5.0) + range(int(np.ceil(N/5.0))) if i < N]
                A_ver = A[indices]
                A_train = np.delete(A, indices, 0)
                y_ver = y[indices].flatten()
                y_train = np.delete(y, indices).flatten()
                
                
                x_train = bp_denoise(A_train, y_train, epsilon[e])
                y_trained = np.reshape(np.dot(A_ver, x_train), len(y_ver))
                
                assert y_trained.shape == y_ver.shape
                errors[n] = np.mean(np.abs(y_trained - y_ver))/len(y_ver)
            
            mean_errors[e] = np.mean(errors)
        
        best_epsilon = epsilon[np.argmin(mean_errors)]
        x = bp_denoise(A, y, best_epsilon)
        residue = np.linalg.norm(np.dot(A, x).flatten() - y.flatten())
        self.coefficients = np.reshape(x, (len(x),1))
    
    def getQuadraturePointsWeights(self, points):
        """
        Generates quadrature points and weights.

        :param Poly self:
            An instance of the Poly class.
        :param int number_of_points:
            Generates a quadrature rule based on the number of points; if an integer value is provided then, QMC is used.
        """
        if points is None:
            points = 4000
        p, w = self.getQuadratureRule(options = self.quadrature_rule, number_of_points = points)
        super(Polycs, self).__setQuadrature__(p,w)

#Generates the projection and preconditoning matrices for the given sampling method.
def samplingMethod(parameters, basis, sampling, no_of_points):
    if not(sampling.lower() in ["standard", "asymptotic", "dlm"]) :
        raise(ValueError, 'Polycs:samplingMethod:: Must supply training x or valid sampling method.') 
    if no_of_points is None:
        no_of_points = int(basis.elements.shape[0]/2)
    dimensions = len(parameters)
    size = (no_of_points, dimensions)  
    p = np.zeros(size, dtype = 'float')
    v = np.zeros((no_of_points, no_of_points), dtype = 'float')
    if sampling.lower() == "standard":
        for i in range(dimensions):
            p[:,i] = parameters[i].getSamples(m=no_of_points).flatten()
        v = np.eye(no_of_points)
    elif sampling.lower() == "asymptotic":
        if not(all([i.param_type.lower() == "uniform" for i in parameters]) or all([i.param_type.lower() == "gaussian" for i in parameters])):
            raise(ValueError, "Polycs:samplingMethod:: Asymptotic sampling only available for uniform and gaussian distribution (for now).")
        if all([i.param_type.lower() == "uniform" for i in parameters]):
            p = np.cos(np.random.uniform(size = p.shape) * np.pi)
            ranges = np.array([param.upper - param.lower for param in parameters], dtype = "float")
            means = np.array([(param.upper + param.lower)/2.0 for param in parameters], dtype = "float")
            
            v = np.diag([np.prod(np.array([(1-p[i,j]**2)**.25 for j in range(dimensions)])) for i in range(no_of_points)])
            p = p * ranges/2.0 + means
        elif all([i.param_type.lower() == "gaussian" for i in parameters]):
            z = np.random.normal(size = p.shape)
            u = np.random.uniform(size = p.shape)
            z_norm = np.linalg.norm(z, axis = 1).reshape((no_of_points,1))
            r = np.sqrt(2 * (2 * max(basis.orders) + 1))
            p = z/z_norm * r * u
            p_norm = np.linalg.norm(p, axis = 1)
            v = np.diag(np.exp(-p_norm**2 /4.0))
    elif sampling.lower() == "dlm":
        orders = np.array([parameters[i].order for i in range(dimensions)])
        wts = np.zeros((dimensions, max(orders) + 1))
        pts = np.zeros((dimensions, max(orders) + 1))
        tensor_grid_size = np.prod(orders+1)
        chosen_points = np.random.choice(range(tensor_grid_size), size = no_of_points, replace = False)
        all_points_index = np.array([i for i in np.ndindex(tuple(orders+1))])
        points_index = all_points_index[chosen_points]
        for u in range(dimensions):
            local_pts, local_wts = parameters[u]._getLocalQuadrature(orders[u])   
            pts[u,:], wts[u,:], = local_pts.flatten(), local_wts.copy()
        for i in range(no_of_points):
            p[i,:] = np.array([pts[j, points_index[i,j]] for j in range(dimensions)])
            v[i,i] = np.prod(np.sqrt(np.array([wts[j, points_index[i,j]] for j in range(dimensions)])))
    x = p.copy()
    w = v.copy()
    return x, w

# Compute coherence of matrix A
def coherence(A):
    norm_A = columnNormalize(A)
    G = np.dot(norm_A.T, norm_A)
    np.fill_diagonal(G, np.nan)
    return np.nanmax(G)




