#!/usr/bin/python
import sys
import numpy as np
from scipy.special import gamma
"""

    The PolynomialParam Class
    Designed to be the base class for all subsequent polynomial / quadrature / optimal quadrature subsampling
    routines. Coding in progress!

    Pranay Seshadri
    ps583@cam.ac.uk

"""
class PolynomialParam(object):
    """ An uncertain parameter.
    Attributes:
        param_type: The distribution associated with the parameter
        lower_bound: Lower bound of the parameter
        upper_bound: Upper bound of the parameter
        shape_parameter_A: Value of the first shape parameter
        shape_parameter_B: Value of the second shape parameter
    """

    """~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                constructor / initializer
       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""

    def __init__(self, param_type, lower_bound, upper_bound, shape_parameter_A, shape_parameter_B, derivative_flag, order):
        """ Return a new uncertain parameter object """
        self.param_type = param_type # string
        self.lower_bound = lower_bound # double
        self.upper_bound = upper_bound # double
        self.shape_parameter_A = shape_parameter_A # double
        self.shape_parameter_B = shape_parameter_B # double
        self.derivative_flag = derivative_flag # integer
        self.order = order # integer
    """~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                get() methods
       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
    def getParamType(self):
        return self.param_type

    def getLowerBound(self):
        return self.lower_bound

    def getUpperBound(self):
        return self.upper_bound

    def getShapeParameterA(self):
        return self.shape_parameter_A

    def getShapeParameterB(self):
        return self.shape_parameter_B

    def getRecurrenceCoefficients(self): # do i really need this!?
        return recurrence_coefficients(self)

    def getJacobiMatrix(self):
        return jacobiMatrix(self)

    def getJacobiEigenvectors(self):
        return jacobiEigenvectors(self)

    def getOrthoPoly(self, points):
        return orthoPolynomial_and_derivative(self, points, derivative_flag)

    def getLocalQuadrature(self):
        return getlocalquadrature(self)

    # Might need another getAmatrix function that doesn't store the full matrix!
    def getAmatrix(self, *argv):

        # If there is an additional argument, then replace the
        for arg in argv:
            self.order = argv[0]
            gridPoints, gridWeights = getlocalquadrature(self, argv[0])
        else:
            gridPoints, gridWeights = getlocalquadrature(self)

        A, C = orthoPolynomial_and_derivative(self, gridPoints)
        A = A.T # Take the temp_transpose
        C = C.T

        return A, C, gridPoints
    """~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    function definitions -- outside PolyParam Class
                    (these are all technically private!)
       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""

# Call different methods depending on the choice of the polynomial parameter
def recurrence_coefficients(self):
    if self.param_type is "Jacobi":
        ab = jacobi_recurrence_coefficients(self.shape_parameter_A, self.shape_parameter_B, self.order)
    elif self.param_type == "Custom":
        ab = custom_recurrence_coefficients(self.lower_bound, self.upper_bound, self.shape_parameter_A, self.shape_parameter_B, self.order)
    elif self.param_type == "Gaussian":
        ab = hermite_recurrence_coefficients(self.lower_bound, self.upper_bound, self.shape_parameter_A, self.shape_parameter_B, self.order)
    return ab

# Recurrence coefficients for Jacobi type parameters
def jacobi_recurrence_coefficients(param_A, param_B, order):

    # Initial setup - check out Walter Gatuschi!
    order = int(order) # check!!
    a0 = (param_B - param_A)/(param_A + param_B + 2.0)
    ab = np.zeros((order,2))
    b2a2 = param_B**2 - param_A**2

    if order > 0 :
        ab[0,0] = a0
        ab[0,1] = 2.0

    for k in range(1,order):
        temp = k + 1
        ab[k,0] = b2a2/((2.0 * (temp - 1) + param_A + param_B) * (2.0 * temp + param_A + param_B))
        if(k == 1):
            ab[k,1] = ( 4.0 * (temp - 1) * (temp - 1 + param_A) * (temp - 1 + param_B)) / ( (2 * (temp - 1) + param_A + param_B  )**2 * (2 * (temp - 1) + param_A + param_B + 1))
        else:
            ab[k,1] = ( 4.0 * (temp - 1) * (temp - 1 + param_A) * (temp - 1 + param_B) * (temp -1 + param_A + param_B) ) / ((2 * (temp - 1) + param_A + param_B)**2 * (2 *(temp -1) + param_A + param_B + 1) * (2 * (temp - 1) + param_A + param_B -1 ) )
    return ab

# Recurrence coefficients for Hermite type parameters
def hermite_recurrence_coefficients(lower, upper, param_A, param_B, order):

    return ab

# Recurrence coefficients for Custom parameters
def custom_recurrence_coefficients(lower, upper, param_A, param_B, order):

    return ab

# Compute the Jacobi matrix. The eigenvalues and eigenvectors of this matrix
# forms the basis of gaussian quadratures
def jacobiMatrix(self):
    order = int(self.order)
    ab = recurrence_coefficients(self)

    # The case of order 1~
    if order == 1:
        JacobiMatrix = ab[0, 0]

    # For everything else~
    else:
        JacobiMatrix = np.zeros((order, order)) # allocate space
        JacobiMatrix[0,0] = ab[0,0]
        JacobiMatrix[0,1] = np.sqrt(ab[1,1])

        for u in range(1,order-1):
            JacobiMatrix[u,u] = ab[u,0]
            JacobiMatrix[u,u-1] = np.sqrt(ab[u,1])
            JacobiMatrix[u,u+1] = np.sqrt(ab[u+1,1])

        JacobiMatrix[order-1, order-1] = ab[order-1,0]
        JacobiMatrix[order-1, order-2] = np.sqrt(ab[order-1,1])

    return JacobiMatrix

# Computes 1D quadrature points and weights
" I think this is scaled to be between [-1,1] by default --> need to change this!"
def getlocalquadrature(self, *argv):

    # If there is an additional argument, then replace the
    for arg in argv:
        self.order = arg

    # Get the recurrence coefficients & the jacobi matrix
    recurrence_coeffs = recurrence_coefficients(self)
    JacobiMat = jacobiMatrix(self)

    # If statement to handle the case where order = 1
    if self.order == 1:
        local_points = [(self.upper_bound - self.lower_bound)/(2.0) + self.lower_bound]
        local_weights = [1.0]
    else:
        # Compute eigenvalues & eigenvectors of Jacobi matrix
        D,V = np.linalg.eig(JacobiMat)
        V = np.mat(V) # convert to matrix
        local_points = np.sort(D) # sort by the eigenvalues
        i = np.argsort(D) # get the sorted indices
        i = np.array(i) # convert to array
        w = np.linspace(1,self.order+1,self.order) # create space for weights
        p = np.ones((self.order,1))
        for u in range(0, len(i) ):
            w[u] = (V[0,i[u]]**2) # replace weights with right value
            p[u,0] = local_points[u]

        local_weights = recurrence_coeffs[0,1] * w # normalizing step
        local_points = p # re-label

    # Return 1D gauss points and weights
    return local_points, local_weights


def jacobiEigenvectors(self):
    JacobiMat = jacobiMatrix(self)
    order = self.order
    if order == 1:
        V = [1.0]
    else:
        D,V = np.linalg.eig(JacobiMat)
        V = np.mat(V) # convert to matrix
        i = np.argsort(D) # get the sorted indices
        i = np.array(i) # convert to array
        V = V[:,i]

    return V


# Univariate orthogonal polynomial correspoding to the weight of the parameter
def orthoPolynomial_and_derivative(self, gridPoints):

    orthopoly = np.zeros((self.order, len(gridPoints))) # create a matrix full of zeros
    derivative_orthopoly = np.zeros((self.order, len(gridPoints)))
    ab = recurrence_coefficients(self)

    # Zeroth order
    orthopoly[0,:] = (1.0)/(1.0 * np.sqrt(ab[0,1]) ) # Correct!

    # Cases
    if self.order == 1:
        return orthopoly

    orthopoly[1,:] = ((gridPoints[:,0] - ab[0,0]) * orthopoly[0,:] ) * (1.0)/(1.0 * np.sqrt(ab[1,1]) )

    if self.order == 2:
        return orthopoly

    if self.order >= 3:
        for u in range(2,self.order):
            # Three-term recurrence rule in action!
            orthopoly[u,:] = ( ((gridPoints[:,0] - ab[u-1,0])*orthopoly[u-1,:]) - np.sqrt(ab[u-1,1])*orthopoly[u-2,:] )/(1.0 * np.sqrt(ab[u,1]))

    # Only if the derivative flag is on do we compute the derivative polynomial
    if self.derivative_flag == 1:
        if self.order == 1:
            return derivative_orthopoly

        derivative_orthopoly[1,:] = orthopoly[0,:] / (np.sqrt(ab[1,1]))

        if self.order == 2:
            return derivative_orthopoly

        if self.order >= 3:
            for u in range(2, self.order):
                # Four-term recurrence formula for derivatives of orthogonal polynomials!
                derivative_orthopoly[u,:] = ( ((gridPoints[:,0] - ab[u-1,0]) * derivative_orthopoly[u-1,:]) - ( np.sqrt(ab[u-1,1]) * derivative_orthopoly[u-2,:] ) +  orthopoly[u-1,:]   )/(1.0 * np.sqrt(ab[u,1]))
        return orthopoly, derivative_orthopoly

    else:
        return orthopoly
