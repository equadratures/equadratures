"""Operations involving polynomial regression on a data set"""
from parameter import Parameter
from basis import Basis
from poly import Poly
import numpy as np
from stats import Statistics, getAllSobol
import scipy


class Polyreg(Poly):
    """
    This class defines a Polyreg (polynomial via regression) object
    :param training_x: A numpy 
    :param IndexSet basis: An instance of the IndexSet class, in case the user wants to overwrite the indices that are obtained using the orders of the univariate parameters in Parameters uq_parameters. The latter corresponds to a tensor grid index set and is the default option if no basis parameter input is given.
    :param parameters: List of instances of Parameters class.
    :param training_y: Column vector (np array) of regression targets corresponding to each row of training_x. Either this or fun should be specified, but not both.
    :param fun: Function to evaluate training_x on to obtain regression targets automatically. Either this or fun should be specified, but not both.
    
    """
    # Constructor
    def __init__(self, parameters, basis, training_x, fun=None, training_y=None):
        super(Polyreg, self).__init__(parameters, basis)
        self.x = training_x
        assert self.x.shape[1] == len(self.parameters) # Check that x is in the correct shape
        if not((training_y is None) ^ (fun is None)):
            raise ValueError("Specify atleast one of fun or training_y.")
        if not(fun is None):
            try:
                self.y = np.apply_along_axis(fun, 1, self.x)
            except:
                raise ValueError("Fun must be callable.")
        else:
            self.y = training_y                           
        if self.dimensions != self.basis.elements.shape[1]:
            raise(ValueError, 'Polyreg:__init__:: The number of parameters and the number of dimensions in the index set must be the same.')
<<<<<<< HEAD
        self.parameters = parameters
        self.A =  getPolynomial(self.parameters, self.scalingX(self.x), self.basis).T
=======
        self.setDesignMatrix()
>>>>>>> 6c602e89c29e644fbc991efe756a706b4a9706c0
        self.cond = np.linalg.cond(self.A)
        self.y = np.reshape(self.y, (len(self.y), 1)) 
        self.computeCoefficients()

<<<<<<< HEAD
    def scalingX(self, x_points_scaled):
        rows, cols = x_points_scaled.shape
        points = np.zeros((rows, cols))
        points[:] = x_points_scaled

        # Now re-scale the points and return only if its not a Gaussian!
        for i in range(0, self.dimensions):
            for j in range(0, rows):
                if (self.parameters[i].param_type == "Uniform"):
                    points[j,i] = 2.0 * ( ( points[j,i] - self.parameters[i].lower) / (self.parameters[i].upper - self.parameters[i].lower) ) - 1.0
                elif (self.parameters[i].param_type == "Beta" ):
                    points[j,i] =  ( points[j,i] - self.parameters[i].lower) / (self.parameters[i].upper - self.parameters[i].lower) 
        
        return points

    def computeActiveSubspaces(self, samples=None):
        d = self.dimensions
        if samples is  None:
            M = 300
            X = np.zeros((M, d))
            for j in range(0, d):
                X[:, j] =  np.reshape( self.parameters[j].getSamples(M), M)
                print np.max(X[:,j]), np.min(X[:,j])
        else:
            X = samples
            M, _ = X.shape
            X = samples

        # Gradient matrix!
        polygrad = self.getPolynomialGradientApproximant(xvalue=X)
        weights = np.ones((M, 1)) / M
        R = polygrad.transpose() * weights
        C = np.dot(polygrad, R )

        # Compute eigendecomposition!
        e, W = np.linalg.eigh(C)
        idx = e.argsort()[::-1]
        eigs = e[idx]
        eigVecs = W[:, idx]
        return eigs, eigVecs


=======
>>>>>>> 6c602e89c29e644fbc991efe756a706b4a9706c0
    # Solve for coefficients using ordinary least squares
    def computeCoefficients(self):
        alpha = np.linalg.lstsq(self.A, self.y) # Opted for numpy's standard version because of speed!
        self.coefficients = alpha[0]
        super(Polyreg, self).__setCoefficients__(self.coefficients)

    def setDesignMatrix(self):
        self.A = self.getPolynomial(self.scalingX(self.x)).T
        super(Polyreg, self).__setDesignMatrix__(self.A)

    def getfitStatistics(self):
        t_stat = get_t_value(self.coefficients, self.A, self.y)
        r_sq = get_R_squared(self.coefficients, self.A, self.y)
        return t_stat, r_sq
<<<<<<< HEAD

    def getStatistics(self, quadratureRule=None):
        p, w = self.getQuadratureRule(quadratureRule)
        evals = getPolynomial(self.parameters, self.scalingX(p), self.basis)
        return Statistics(self.coefficients, self.basis, self.parameters, p, w, evals)

    def getPolynomialApproximant(self):
        return self.A * np.mat(self.coefficients)

    def getPolynomialGradientApproximant(self, xvalue=None):
        if xvalue is None:
            xvalue = self.x
        H = getPolynomialGradient(self.parameters, self.scalingX(xvalue), self.basis)
        grads = np.zeros((self.dimensions, len(xvalue) ) )
        for i in range(0, self.dimensions):
            grads[i,:] = np.mat(self.coefficients).T * H[i]
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
#            print p.shape
            return p,w
=======
>>>>>>> 6c602e89c29e644fbc991efe756a706b4a9706c0
    
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
    # Get points of polynomial basis functions on the points specified by x
    # basically just calls getPolynomial with scaled points
    # rows = number of points, cols = number of dimensions
    def getPolynomial_x(self, x):
        return getPolynomial(self.parameters, self.scalingX(x) , self.basis)
=======
>>>>>>> 6c602e89c29e644fbc991efe756a706b4a9706c0
    
def get_t_value(coefficients, A, y):
    RSS = np.linalg.norm(y - np.dot(A,coefficients))**2
    n,p = A.shape
    if n == p:
        return "exact"
    RSE = RSS/(n-p)
    Q, R = np.linalg.qr(A)
    inv_ATA = np.linalg.inv(np.dot(R.T, R))
    se = np.array([np.sqrt(RSE * inv_ATA[j,j]) for j in range(p)])
    t_stat = coefficients / np.reshape(se, (len(se), 1))
    # p-value is scipy.stats.t.cdf(t_stat, n - p)
    return t_stat

<<<<<<< HEAD
def getTensorQuadratureRule(stackOfParameters, dimensions, orders):
        flag = 0

        # Initialize points and weights
        pp = [1.0]
        ww = [1.0]

        # number of parameters
        # For loop across each dimension
        for u in range(0, dimensions):

            # Call to get local quadrature method (for dimension 'u')
            local_points, local_weights = stackOfParameters[u]._getLocalQuadrature(orders[u], scale=True)
            
#            print local_points
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

=======
>>>>>>> 6c602e89c29e644fbc991efe756a706b4a9706c0
def get_R_squared(alpha, A, y):
    y_bar = scipy.mean(y) * np.ones(len(y))
    TSS = np.linalg.norm(y - y_bar)**2
    RSS = np.linalg.norm(np.dot(A,alpha) - y)**2
    return 1 - RSS/TSS

<<<<<<< HEAD
def getPolynomial(stackOfParameters, stackOfPoints, chosenBasis):
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

def getPolynomialGradient(stackOfParameters, stackOfPoints, chosenBasis):
     # "Unpack" parameters from "self"
    basis = chosenBasis.elements
    basis_entries, dimensions = basis.shape
    no_of_points, _ = stackOfPoints.shape
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
=======
>>>>>>> 6c602e89c29e644fbc991efe756a706b4a9706c0
