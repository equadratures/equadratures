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
        self.setDesignMatrix()
        self.cond = np.linalg.cond(self.A)
        self.y = np.reshape(self.y, (len(self.y), 1)) 
        self.computeCoefficients()

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

