"""Operations involving polynomial regression on a data set"""
from parameter import Parameter
from indexset import IndexSet
import numpy as np
from math import factorial
from itertools import combinations
from utils import evalfunction, find_repeated_elements, meshgrid
from plotting import bestfit, bestfit3D, histogram
from qr import solveLSQ
from stats import Statistics

class Polyreg(object):
    """
    This class defines a Polyreg (polynomial via regression) object

    :param training_x: A numpy 
    :param IndexSet index_set: An instance of the IndexSet class, in case the user wants to overwrite the indices that are obtained using the orders of the univariate parameters in Parameters uq_parameters. The latter corresponds to a tensor grid index set and is the default option if no index_set parameter input is given.
    
    """
    # Constructor
    def __init__(self, training_x, training_y, option):
        self.training_x = training_x
        self.training_y = training_y
        self.option = option
        X = self.training_x
        Y = self.training_y

        m, n = X.shape
        ones = np.ones((m, 1))
        self.dimensions = n
        #total_terms = nchoosek(n) + order, order)        

        if self.option is 'linear':
            A = np.mat(np.hstack([X, ones]), dtype='float64')
            coeffs, not_used = solveLSQ(A, Y)
            self.coefficients =  np.mat(coeffs, dtype='float64')
            self.Xmatrix = A

        elif self.option is 'quadratic':
           dimensions = n
           variables = range(0, dimensions)
           combination = list(combinations(variables, 2)) 
           constants = np.mat(np.ones((m, 1)), dtype='float64')

            # Compute the interaction terms!
           XC = np.mat( np.ones((m, len(combination))) , dtype='float64')
           for i in range(0, len(combination) ):
                for j in range(0, m):
                    XC[j,i] = X[j, combination[i][0] ] * X[j, combination[i][1] ] ; 

           # Compute the squared terms
           X2 = np.mat(np.ones((m, dimensions ) ) , dtype = 'float64')
           for i in range(0, dimensions ):
                for j in range(0, m):
                    X2[j,i] = X[j, i] * X[j,i ] ; 

           # Set up the A matrix
           A = np.mat(np.hstack([constants, X, X2, XC]), dtype = 'float64' )
           self.Xmatrix = A
           coeffs, not_used = solveLSQ(A, Y)
           self.coefficients = np.mat(coeffs, dtype='float64')
        else:
            raise(ValueError, 'PolyFit.__init__: invalid fitting option: Choose between linear or quadratic.')

    # Test Polynomial
    def testPolynomial(self, test_x):
        """
        Returns the PDF of the model output. This routine effectively multiplies the coefficients of a polynomial
        expansion with its corresponding basis polynomials. 
    
        :param PolyFit self: An instance of the PolyFit class
        :param: numpy-matrix test_x: The function that needs to be approximated (or interpolated)
        :return: polyapprox: The polynomial expansion of a function
        :rtype: numpy matrix

        """
        coefficients = self.coefficients
        p, q = test_x.shape
        m, dimensions = self.training_x.shape
        if self.option is 'linear':
            m = len(coefficients) - 1
            constant_term = coefficients[m]
            linear_terms = np.mat(coefficients[0:m], dtype='float64')
            test_y = np.mat(np.zeros((p, 1)) , dtype='float64' )

            # add a for loop here!
            for i in range(0, p):
                test_y[i,0] = (linear_terms.T * test_x[i,:].T) + constant_term
            return test_y

        elif self.option is 'quadratic':
            if dimensions == 1:
                test_y = np.mat(np.zeros((p, 1)) , dtype='float64' )
                for i in range(0, p):
                    test_y[i,0] = (self.coefficients[2] * test_x[i,0]**2) + (self.coefficients[1] * test_x[i,:].T) + self.coefficients[0]
                return test_y
            else:
                variables = range(0, dimensions)
                A = np.mat( np.zeros((dimensions, dimensions)), dtype='float64')
                c = np.mat( np.zeros((dimensions, 1)), dtype='float64') 
                combination = list(combinations(variables, 2))
            
                # For the interaction terms!
                for i in range(0, dimensions):
                    for j in range(0, dimensions):
                        if j < i :
                            for k in range(0, len(combination)):
                                if (combination[k][0] == i and combination[k][1] == j) or (combination[k][1] == i and combination[k][0] == j ) : 
                                    entry = k
                            A[i, j] = self.coefficients[dimensions*2 + entry] * 0.5
                            A[j, i] = A[i, j] # Because A is a symmetric matrix!
                    A[i,i] = self.coefficients[i + (2*dimensions - 1)] # Diagonal elements of A -- which house the quadratic terms!
                
                # For the linear terms!
                for i in range(0, dimensions):
                    c[i] = self.coefficients[i+1]        
                d = self.coefficients[0] # constant term!

                test_y = np.mat(np.zeros((p, 1)) , dtype='float64' )
                for i in range(0, p):
                    test_y[i,0] = (test_x[i,:] * A * test_x[i,:].T) + (c.T * test_x[i,:].T) + d
                return test_y
        
    def plot(self, test_x, filename=None):
        """
        Returns the PDF of the model output. This routine effectively multiplies the coefficients of a polynomial
        expansion with its corresponding basis polynomials. 
    
        :param Polynomial self: An instance of the Polynomial class
        :param: callable function: The function that needs to be approximated (or interpolated)
        :return: polyapprox: The polynomial expansion of a function
        :rtype: numpy matrix

        """
        dimensions = self.dimensions
        if dimensions == 1:
            xx = np.mat(test_x, dtype='float64')
            test_x = xx.T
            test_y = self.testPolynomial(test_x)
            N = len(self.training_y) # number of training points!

            if self.option is 'linear':
                X = self.Xmatrix
                w = np.linalg.inv(X.T * X) * X.T * self.training_y
                m, n = test_x.shape
                ones = np.ones((m, 1))
                test_X = np.mat(np.hstack([test_x, ones]), dtype='float64')
                test_mean = test_X * w 
                ss = 1.0/(1.0 * N) * ( self.training_y.T * self.training_y - self.training_y.T * X * w)
                test_var = ss * np.diag(test_X * np.linalg.inv(X.T * X) * test_X.T)
                bestfit(self.training_x, self.training_y, test_x, test_y, test_var.T, r'$X$', r'$Y$', filename)

            elif self.option is 'quadratic':
                X = self.Xmatrix
                w = np.linalg.inv(X.T * X) * X.T * self.training_y
                m, n = test_x.shape
                ones = np.ones((m, 1))
                squares_test_x = np.mat(np.zeros((m , n)), dtype='float64')
                for i in range(0, m):
                    squares_test_x[i] = test_x[i]**2
                test_X = np.mat(np.hstack([ones, test_x, squares_test_x]), dtype='float64')
                test_mean = test_X * w 
                ss = 1.0/(1.0 * N) * ( self.training_y.T * self.training_y - self.training_y.T * X * w)
                test_var = ss * np.diag(test_X * np.linalg.inv(X.T * X) * test_X.T)
                bestfit(self.training_x, self.training_y, test_x, test_y, test_var.T, r'$X$', r'$Y$', filename)

        elif dimensions == 2:
            X1 = test_x[0]
            X2 = test_x[1]
            xx1, xx2 = np.meshgrid(X1, X2)
            u, v = xx1.shape
            test_x = np.mat( np.hstack( [np.reshape(xx1, (u*v, 1)),  np.reshape(xx2, (u*v, 1)) ]) , dtype='float64')
            test_y = self.testPolynomial(test_x)
            yy1 = np.reshape(test_y, (u, v))
            bestfit3D(self.training_x, self.training_y, [xx1, xx2], yy1, r'$X_1$', r'$X_2$', r'$Y$', filename)


# Routine for computing n choose k
def nchoosek(n, k):
    numerator = factorial(n)
    denominator = factorial(k) * factorial(n - k)
    return (1.0 * numerator) / (1.0 * denominator)


