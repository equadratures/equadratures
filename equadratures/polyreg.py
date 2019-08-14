"""Finding coefficients via regression."""
from parameter import Parameter
from basis import Basis
from poly import Poly
import numpy as np
from stats import Statistics, getAllSobol
import scipy


class Polyreg(Poly):
    """
    The class defines a Polyreg object. It is the child of Poly.
    :param Parameter parameters:
        A list of parameters.
    :param Basis basis:
        A basis selected for the multivariate polynomial.
    :param matrix training_inputs:
        A N-by-d matrix of input training points.
    :param matrix training_outputs:
        A N-by-1 matrix of output training points.
    :param callable fun:
        Instead of specifying the output training points, the user can also provide a callable function, which will be evaluated.
    """
    # Constructor
    def __init__(self, parameters, basis, training_inputs = None, fun=None, training_outputs=None, quadrature_rule = None, no_of_quad_points = None):
        super(Polyreg, self).__init__(parameters, basis)
        if not(training_inputs is None):
            self.x = training_inputs
            assert self.x.shape[1] == len(self.parameters) # Check that x is in the correct shape

        if not((training_outputs is None) ^ (fun is None)):
            raise ValueError("Specify atleast one of fun or training_outputs.")
        if not(fun is None):
            try:
                self.y = np.apply_along_axis(fun, 1, self.x)
            except:
                raise ValueError("Fun must be callable.")
        else:
            self.y = training_outputs
        if self.dimensions != self.basis.elements.shape[1]:
            raise(ValueError, 'Polyreg:__init__:: The number of parameters and the number of dimensions in the index set must be the same.')
        self.setDesignMatrix()
        self.cond = np.linalg.cond(self.A)
        self.y = np.reshape(self.y, (len(self.y), 1))
        self.computeCoefficients()
        self.quadrature_rule = quadrature_rule
        self.getQuadraturePointsWeights(no_of_quad_points)

    # Solve for coefficients using ordinary least squares
    def computeCoefficients(self):
        """
        This function computes the coefficients using least squares. To access the coefficients simply use the class's attribute self.coefficients.
        :param Polyreg self:
            An instance of the Polyreg class.
        """
        p = len(self.y)
        self.bz = np.dot( self.Wz ,  np.reshape(self.y, (p,1)) )
        alpha = np.linalg.lstsq(self.A, self.bz) # Opted for numpy's standard version because of speed!
        self.coefficients = alpha[0]
        super(Polyreg, self).__setCoefficients__(self.coefficients)

    def setDesignMatrix(self):
        """
        Sets the design matrix using the polynomials defined in the basis.
        :param Polyreg self:
            An instance of the Polyreg class.
        """
        Pz = super(Polyreg, self).getPolynomial(self.x)
        wts =  1.0/(np.sum( Pz**2 , 0)**2)
        wts = wts * 1.0/np.sum(wts)
        

        #wts =  1.0/(np.sum( super(Polyreg, self).getPolynomial(self.x)**2 , 0) )**2
        #wts = wts * 1.0/np.sum(wts)
        self.Wz = np.mat(np.diag( np.sqrt(wts) ) )
        self.A =  self.Wz * Pz.T
        rows, cols = self.A.shape
        if rows <= cols:
            raise(ValueError, 'Polyreg:setDesignMatrix:: Number of columns have to be less than (or equal to) the number of rows!')
        super(Polyreg, self).__setDesignMatrix__(self.A)

    def getfitStatistics(self):
        """
        Computes statistics based on the quality of the fit
        :param Polyreg self:
            An instance of the Polyreg class.
        :return:
            `T statistic <https://en.wikipedia.org/wiki/T-statistic>`_.
        :return:
            `Coefficient of determination / R-squared value <https://en.wikipedia.org/wiki/Coefficient_of_determination>`_.
        """
        t_stat = get_t_value(self.coefficients, self.A, self.y)
        r_sq = get_R_squared(self.coefficients, self.A, self.y)
        return t_stat, r_sq
    
    def getQuadraturePointsWeights(self, points):
        if points is None:
            points = 10000
        p, w = self.getQuadratureRule(options = self.quadrature_rule, number_of_points = points)
        super(Polyreg, self).__setQuadrature__(p,w)

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

def get_R_squared(alpha, A, y):
    y_bar = scipy.mean(y) * np.ones(len(y))
    TSS = np.linalg.norm(y - y_bar)**2
    RSS = np.linalg.norm(np.dot(A,alpha) - y)**2
    return 1 - RSS/TSS
