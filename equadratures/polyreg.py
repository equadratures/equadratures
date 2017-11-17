"""Operations involving polynomial regression on a data set"""
from .parameter import Parameter
from .indexset import IndexSet
import numpy as np
import scipy
from math import factorial
from itertools import combinations
from .utils import evalfunction, find_repeated_elements, meshgrid
from .plotting import bestfit, bestfit3D, histogram
from .qr import solveLSQ, qr_MGS
from .stats import Statistics

class Polyreg(object):
    """
    This class defines a Polyreg (polynomial via regression) object

    :param training_x: A numpy 
    :param IndexSet index_set: An instance of the IndexSet class, in case the user wants to overwrite the indices that are obtained using the orders of the univariate parameters in Parameters uq_parameters. The latter corresponds to a tensor grid index set and is the default option if no index_set parameter input is given.
    :param parameters: List of instances of Parameters class.
    :param training_y: Column vector (np array) of regression targets corresponding to each row of training_x. Either this or fun should be specified, but not both.
    :param fun: Function to evaluate training_x on to obtain regression targets automatically. Either this or fun should be specified, but not both.
    
    """
    # Constructor
    def __init__(self, training_x, parameters, index_set, fun = None, training_y = None):
        self.x = training_x
        #Check that x is in the correct shape
        assert self.x.shape[1] == len(parameters)
        
        if not((training_y is None) ^ (fun is None)):
            raise ValueError("Specify only one of fun or training_y.")
        
        if not(fun is None):
            try:
                self.y = np.apply_along_axis(fun, 1, self.x)
            except:
                raise ValueError("Fun must be callable.")
        else:
            self.y = training_y                    
                
                
        self.index_set = index_set
        self.parameters = parameters
        
        self.A = getMultivariatePolynomial(parameters, self.x, index_set).T
        # Make sure y is a column vector
        self.y = np.reshape(self.y, (len(self.y), 1)) 
    
    # Solve for coefficients using ordinary least squares
    def OLS(self):
        alpha, cond = solveLSQ(self.A, self.y)
        t_stat = get_t_value(alpha, self.A, self.y)
        r_sq = get_R_squared(alpha, self.A, self.y)
        return alpha, cond, t_stat, r_sq
        
#TODO: t stat, R square
#TODO: test hypothesis for given subset specified by given index set

# Get t value for each "variable" (basis term in this case) given regression parameters
def get_t_value(alpha, A, y):
    RSS = np.linalg.norm(y - np.dot(A,alpha))**2
    p = A.shape[1]
    n = A.shape[0]
    if n == p:
        return "exact"
    RSE = RSS/(n-p)
    
    Q, R = qr_MGS(A)
    inv_ATA = np.linalg.inv(np.dot(R.T, R))
    se = np.array([np.sqrt(RSE * inv_ATA[j,j]) for j in range(p)])
    
    t_stat = alpha / np.reshape(se, (len(se), 1))
    return t_stat
    
def get_R_squared(alpha, A, y):
    y_bar = scipy.mean(y) * np.ones(len(y))
    TSS = np.linalg.norm(y - y_bar)**2
    RSS = np.linalg.norm(np.dot(A,alpha) - y)**2
    return 1 - RSS/TSS

# Copied from polyint, removed derivative evaluation
def getMultivariatePolynomial(stackOfParameters, stackOfPoints, indexsets=None):
    # "Unpack" parameters from "self"
    isets = indexsets
    if indexsets is None:
        if isets.index_set_type == 'Sparse grid':
            ic, not_used, index_set = isets.getIndexSet()
        else:
            index_set = isets.elements
    else:
        index_set = indexsets.elements

    dimensions = len(stackOfParameters)
    p = {}

    # Save time by returning if univariate!
    if dimensions == 1:
        poly , _ =  stackOfParameters[0].getOrthoPoly(stackOfPoints)
        return poly
    else:
        for i in range(0, dimensions):
            if len(stackOfPoints.shape) == 1:
                stackOfPoints = np.array([stackOfPoints])
            assert len(stackOfPoints.shape) == 2
            p[i] , _ = stackOfParameters[i].getOrthoPoly(stackOfPoints[:,i], int(np.max(index_set[:,i] + 1) ) )

    # Now we multiply components according to the index set
    no_of_points = len(stackOfPoints)
    polynomial = np.zeros((len(index_set), no_of_points))

    # One loop for polynomials
    for i in range(0, len(index_set)):
        temp = np.ones((1, no_of_points))
        for k in range(0, dimensions):
            polynomial[i,:] = p[k][int(index_set[i,k])] * temp
            temp = polynomial[i,:]
    return polynomial