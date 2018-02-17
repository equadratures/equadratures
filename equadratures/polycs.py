<<<<<<< HEAD
"""Finding coefficients via compressive sensing"""
=======
"""Operations involving polynomial regression on a data set"""
>>>>>>> 6c602e89c29e644fbc991efe756a706b4a9706c0
from parameter import Parameter
from basis import Basis
import numpy as np
#from qr import solveLSQ, qr_MGS
from stats import Statistics, getAllSobol
from convex import *
import scipy
<<<<<<< HEAD
# Or do with vanilla numpy?
from sklearn.preprocessing import normalize
=======
>>>>>>> 6c602e89c29e644fbc991efe756a706b4a9706c0


#TODO: Compute coherence, examine sampling methods

class Polycs(object):
    """
    This class defines a Polycs (polynomial via compressive sensing) object
    :param training_x: A numpy 
    :param IndexSet basis: An instance of the IndexSet class, in case the user wants to overwrite the indices that are obtained using the orders of the univariate parameters in Parameters uq_parameters. The latter corresponds to a tensor grid index set and is the default option if no basis parameter input is given.
    :param parameters: List of instances of Parameters class.
    :param training_y: Column vector (np array) of regression targets corresponding to each row of training_x. Either this or fun should be specified, but not both.
    :param fun: Function to evaluate training_x on to obtain regression targets automatically. Either this or fun should be specified, but not both.
    
    """
    # Constructor
<<<<<<< HEAD
    def __init__(self, parameters, basis, training_x = None, sampling = None, no_of_points = None, fun=None, training_y=None):
        self.basis = basis
        self.dimensions = len(parameters)
        self.parameters = parameters
        
        if not(training_x is None):
            self.x = training_x
            assert self.x.shape[1] == len(parameters) # Check that x is in the correct shape
            w = np.eye(self.x.shape[0])
        else:
            self.x, w = self.sample_X(self.parameters, self.basis, sampling, no_of_points)
=======
    def __init__(self, training_x, parameters, basis, fun=None, training_y=None):
        self.x = training_x
        assert self.x.shape[1] == len(parameters) # Check that x is in the correct shape
>>>>>>> 6c602e89c29e644fbc991efe756a706b4a9706c0
        if not((training_y is None) ^ (fun is None)):
            raise ValueError("Specify only one of fun or training_y.")
        if not(fun is None):
            try:
<<<<<<< HEAD
                
                self.y = np.dot(w, np.apply_along_axis(fun, 1, self.x))
                
            except:
                raise ValueError("Fun must be callable.")
        else:
            self.y = np.dot(w, training_y)
        
        if self.dimensions != self.basis.elements.shape[1]:
            raise(ValueError, 'Polycs:__init__:: The number of parameters and the number of dimensions in the index set must be the same.')
        
        self.A =  getPolynomial(self.parameters, self.scalingX(self.x), self.basis).T
        self.A = np.dot(w, self.A)
        print self.A.shape
        print "coherence"
        print coherence(self.A)
        self.cond = np.linalg.cond(self.A)
        self.y = np.reshape(self.y, (len(self.y), 1)) 
        self.computeCoefficients()
    
    @staticmethod
    def sample_X(parameters, basis, sampling, no_of_points):
        
        if not(sampling.lower() in ["standard", "asymptotic", "dlm"]) :
            raise(ValueError, 'Polycs:sample_X:: Must supply training x or valid sampling method.') #Change
        
        if no_of_points is None:
            no_of_points = int(basis.elements.shape[0]/2)
        dimensions = len(parameters)
        size = (no_of_points, dimensions)  
        p = np.zeros(size, dtype = 'float')
        v = np.zeros((no_of_points, no_of_points), dtype = 'float')
#        print sampling.lower()
        if sampling.lower() == "standard":
            for i in range(dimensions):

                p[:,i] = parameters[i].getSamples(m=no_of_points).flatten()
            v = np.eye(no_of_points)
        elif sampling.lower() == "asymptotic":
            if not(all([i.param_type.lower() == "uniform" for i in parameters]) or all([i.param_type.lower() == "gaussian" for i in parameters])):
                raise(ValueError, "Polycs:sample_X:: Asymptotic sampling only available for uniform and gaussian distribution (for now).")
            if all([i.param_type.lower() == "uniform" for i in parameters]):
                p = np.cos(np.random.uniform(size = p.shape) * np.pi)
                ranges = np.array([param.upper - param.lower for param in parameters], dtype = "float")
#                ranges = np.vstack([one_d_ranges for i in range(no_of_points)])
                means = np.array([(param.upper + param.lower)/2.0 for param in parameters], dtype = "float")
#                means = np.vstack([one_d_means for i in range(no_of_points)])
                
                p = p * ranges/2.0 + means
                v = np.diag([np.prod(np.array([(1-p[i,j]**2)**.25 for j in range(dimensions)])) for i in range(no_of_points)])
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
                local_pts, local_wts = parameters[u]._getLocalQuadrature(orders[u], scale=True)
                
                pts[u,:], wts[u,:], = local_pts.flatten(), local_wts.copy()
#            print pts
#            print wts
            for i in range(no_of_points):
                p[i,:] = np.array([pts[j, points_index[i,j]] for j in range(dimensions)])
                v[i,i] = np.prod(np.sqrt(np.array([wts[j, points_index[i,j]] for j in range(dimensions)])))
        
#        print p[:10,:]
#        print v[:10,:]
        x = p.copy()
        w = v.copy()
        return x, w
    
=======
                self.y = np.apply_along_axis(fun, 1, self.x)
            except:
                raise ValueError("Fun must be callable.")
        else:
            self.y = training_y                           
        self.basis = basis
        self.dimensions = len(parameters)
        if self.dimensions != self.basis.elements.shape[1]:
            raise(ValueError, 'Polyreg:__init__:: The number of parameters and the number of dimensions in the index set must be the same.')
        self.parameters = parameters
        self.A =  getPolynomial(self.parameters, self.scalingX(self.x), self.basis).T
        print self.A.shape
        self.cond = np.linalg.cond(self.A)
        self.y = np.reshape(self.y, (len(self.y), 1)) 
        self.computeCoefficients()

>>>>>>> 6c602e89c29e644fbc991efe756a706b4a9706c0
    #def computeWeights(self):
    #    self.weights = 1.0
        #for i in range(0, self.dimensions):
        #    if (self.parameters[i].param_type == "Uniform"):
        #        self.weights = self.weights * (self.parameters[i].upper - self.parameters[i].lower) / (2.0)
        #    elif (self.parameters[i].param_type == "Beta" ):
        #        self.weights = 1.0 / (self.parameters[i].upper - self.parameters[i].lower) 

    def scalingX(self, x_points_scaled):
        rows, cols = x_points_scaled.shape
        points = np.zeros((rows, cols))
        points[:] = x_points_scaled

        # Now re-scale the points and return only if its not a Gaussian!
        for i in range(0, self.dimensions):
            for j in range(0, rows):
                if (self.parameters[i].param_type == "Uniform"):
                    #print points[j,i]
                    points[j,i] = 2.0 * ( ( points[j,i] - self.parameters[i].lower) / (self.parameters[i].upper - self.parameters[i].lower) ) - 1.0
                    #print points[j,i]
                    #print '--------'
                elif (self.parameters[i].param_type == "Beta" ):
                    points[j,i] =  ( points[j,i] - self.parameters[i].lower) / (self.parameters[i].upper - self.parameters[i].lower) 
        
        return points


    # Solve for coefficients using ordinary least squares
    def computeCoefficients(self):
        # Partition data
        A = self.A
        y = self.y.flatten()
        N = A.shape[0]
        # Possible noise levels
        log_epsilon = [-8,-7,-6,-5, -4, -3, -2, -1, 0]
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
            
<<<<<<< HEAD
#            print "noise"
#            print epsilon[e]
#            print "errors"
#            print errors
#            print "mean error"
#            print np.mean(errors)
=======
            print "noise"
            print epsilon[e]
            print "errors"
            print errors
            print "mean error"
            print np.mean(errors)
>>>>>>> 6c602e89c29e644fbc991efe756a706b4a9706c0
            
            mean_errors[e] = np.mean(errors)
        
        best_epsilon = epsilon[np.argmin(mean_errors)]
<<<<<<< HEAD
#        print "best epsilon"
#        print best_epsilon
=======
        print "best epsilon"
        print best_epsilon
>>>>>>> 6c602e89c29e644fbc991efe756a706b4a9706c0
        x = bp_denoise(A, y, best_epsilon)
        
        #Calculate residue
        residue = np.linalg.norm(np.dot(A, x).flatten() - y.flatten())
        print "-----"
        print "overall residue (Should be 0!)"
        print residue
        print "-----"
        self.coefficients = np.reshape(x, (len(x),1))

    def getfitStatistics(self):
        t_stat = get_t_value(self.coefficients, self.A, self.y)
        r_sq = get_R_squared(self.coefficients, self.A, self.y)
        return t_stat, r_sq

    def getStatistics(self, quadratureRule=None):
        p, w = self.getQuadratureRule(quadratureRule)
        evals = getPolynomial(self.parameters, self.scalingX(p), self.basis)
        return Statistics(self.coefficients, self.basis, self.parameters, p, w, evals)

    def getPolynomialApproximant(self):
        return self.A * np.mat(self.coefficients)
    
    def getPolynomialGradientApproximant(self, direction=None, xvalue=None):
        if xvalue is None:
            xvalue = self.x
        
        if direction is not None:
            C = getPolynomialGradient(self.parameters, self.scalingX(xvalue), self.basis, direction).T
            return C * np.mat(self.coefficients)
        else:
            grads = np.zeros((self.dimensions, len(xvalue) ) )
            for i in range(0, self.dimensions):
                v = getPolynomialGradient(self.parameters, self.scalingX(xvalue), self.basis, i).T * np.mat(self.coefficients)
                grads[i, :] = v.reshape((len(xvalue), ))
            return grads
    
    def getPolyFit(self):
        return lambda (x): getPolynomial(self.parameters, self.scalingX(x) , self.basis).T *  np.mat(self.coefficients)
    
    def getPolynomial_t(self, x):
        return getPolynomial(self.parameters, self.scalingX(x) , self.basis).T *  np.mat(self.coefficients)
    
    def getPolyGradFit(self):
        return lambda (x) : self.getPolynomialGradientApproximant(xvalue=x)

    def getQuadratureRule(self, options=None):
        if options is None:
            if self.dimensions > 8:
                options = 'qmc'
            elif self.dimensions < 8 :
                options = 'tensor grid'
        
        options = 'tensor grid'
        if options.lower() == 'qmc':
            default_number_of_points = 20000
            p = np.zeros((default_number_of_points, self.dimensions)) 
            w = 1.0/float(default_number_of_points) * np.ones((default_number_of_points))
            for i in range(0, self.dimensions):
                p[:,i] = self.parameters[i].getSamples(m=default_number_of_points).reshape((default_number_of_points,))
            
            return p, w
        if options.lower() == 'tensor grid':
            
            p,w = getTensorQuadratureRule(self.parameters, self.dimensions, [2*i for i in self.basis.orders])
            
            return p,w
    
    @staticmethod
    def get_F_stat(coefficients_0, A_0, coefficients_1, A_1, y):
        assert len(coefficients_0) != len(coefficients_1)
        assert A_0.shape[0] == A_1.shape[0]
        # Set 0 to be reduced model, 1 to be "full" model
        if len(coefficients_0) > len(coefficients_1):
            temp = coefficients_0.copy()
            coefficients_0 = coefficients_1.copy()
            coefficients_1 = temp
        assert len(coefficients_0) < len(coefficients_1)
        
        RSS_0 = np.linalg.norm(y - np.dot(A_0,coefficients_0))**2
        RSS_1 = np.linalg.norm(y - np.dot(A_1,coefficients_1))**2
        
        n = A_0.shape[0]
        p_1 = A_1.shape[1]
        p_0 = A_0.shape[1]
        F = (RSS_0 - RSS_1) * (n-p_1)/(RSS_1 * (p_1 - p_0))
        # p-value is scipy.stats.f.cdf(F, n - p_1, p_1 - p_0)
        return F

<<<<<<< HEAD
# Compute coherence of matrix A
def coherence(A):
    norm_A = normalize(A, axis = 0)
    G = np.dot(norm_A.T, norm_A)
    np.fill_diagonal(G, np.nan)
    return np.nanmax(G)









=======
>>>>>>> 6c602e89c29e644fbc991efe756a706b4a9706c0
def get_t_value(coefficients, A, y):
    RSS = np.linalg.norm(y - np.dot(A,coefficients))**2
    p, n = A.shape
    if n == p:
        return "exact"
    RSE = RSS/(n-p)
    Q, R = np.linalg.qr(A)
    inv_ATA = np.linalg.inv(np.dot(R.T, R))
    se = np.array([np.sqrt(RSE * inv_ATA[j,j]) for j in range(p)])
    t_stat = coefficients / np.reshape(se, (len(se), 1))
    # p-value is scipy.stats.t.cdf(t_stat, n - p)
    return t_stat


    


def getTensorQuadratureRule(stackOfParameters, dimensions, orders):
        flag = 0

        # Initialize points and weights
        pp = [1.0]
        ww = [1.0]

        # number of parameters
        # For loop across each dimension
        for u in range(0, dimensions):

            # Call to get local quadrature method (for dimension 'u')
<<<<<<< HEAD
            local_points, local_weights = stackOfParameters[u]._getLocalQuadrature(orders[u], scale=True)
#            print local_points
=======
            local_points, local_weights = stackOfParameters[u].getLocalQuadrature(orders[u], scale=True)

>>>>>>> 6c602e89c29e644fbc991efe756a706b4a9706c0
            # Tensor product of the weights
            ww = np.kron(ww, local_weights)

            # Tensor product of the points
            dummy_vec = np.ones((len(local_points), 1))
            dummy_vec2 = np.ones((len(pp), 1))
            left_side = np.array(np.kron(pp, dummy_vec))
            right_side = np.array( np.kron(dummy_vec2, local_points) )
            pp = np.concatenate((left_side, right_side), axis = 1)

        # Ignore the first column of pp
        points = pp[:,1::]
        weights = ww

        # Return tensor grid quad-points and weights
        return points, weights



def get_R_squared(alpha, A, y):
    y_bar = scipy.mean(y) * np.ones(len(y))
    TSS = np.linalg.norm(y - y_bar)**2
    RSS = np.linalg.norm(np.dot(A,alpha) - y)**2
    return 1 - RSS/TSS

def getPolynomial(stackOfParameters, stackOfPoints, chosenBasis):
    #print stackOfPoints
    #return 0
    # "Unpack" parameters from "self"
    basis = chosenBasis.elements
    basis_entries, dimensions = basis.shape
    no_of_points, _ = stackOfPoints.shape
    polynomial = np.zeros((basis_entries, no_of_points))
    p = {}

    # Save time by returning if univariate!
    if dimensions == 1:
        poly , _ =  stackOfParameters[0]._getOrthoPoly(stackOfPoints, int(np.max(basis)))
        return poly
    else:
        for i in range(0, dimensions):
            if len(stackOfPoints.shape) == 1:
                stackOfPoints = np.array([stackOfPoints])
            p[i] , _ = stackOfParameters[i]._getOrthoPoly(stackOfPoints[:,i], int(np.max(basis[:,i])) )

    # One loop for polynomials
    for i in range(0, basis_entries):
        temp = np.ones((1, no_of_points))
        for k in range(0, dimensions):
            polynomial[i,:] = p[k][int(basis[i,k])] * temp
            temp = polynomial[i,:]
    
    return polynomial

def getPolynomialGradient(stackOfParameters, stackOfPoints, chosenBasis, gradDirection):
     # "Unpack" parameters from "self"
    basis = chosenBasis.elements
    basis_entries, dimensions = basis.shape
    no_of_points, _ = stackOfPoints.shape
    polynomialgradient = np.zeros((basis_entries, no_of_points))
    p = {}
    dp = {}

    # Save time by returning if univariate!
    if dimensions == 1:
        poly , _ =  stackOfParameters[0]._getOrthoPoly(stackOfPoints)
        return poly
    else:
        for i in range(0, dimensions):
            if len(stackOfPoints.shape) == 1:
                stackOfPoints = np.array([stackOfPoints])
            p[i] , dp[i] = stackOfParameters[i]._getOrthoPoly(stackOfPoints[:,i], int(np.max(basis[:,i])) )

    # One loop for polynomials
    R = []
    for v in range(0, dimensions):
        gradDirection = v
        polynomialgradient = np.zeros((basis_entries, no_of_points))
        for i in range(0, basis_entries):
            temp = np.ones((1, no_of_points))
            for k in range(0, dimensions):
                if k == gradDirection:
                    polynomialgradient[i,:] = dp[k][int(basis[i,k])] * temp
                else:
                    polynomialgradient[i,:] = p[k][int(basis[i,k])] * temp
                temp = polynomialgradient[i,:]
        R.append(polynomialgradient)

    return R
    
