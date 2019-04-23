"Functionalities for many parameters."
import numpy as np
from scipy import optimize
from .parameter import Parameter
from .polyint import Polyint 
from .basis import Basis 
from scipy import stats

#import matplotlib.pyplot as plt

class Manyparameters(object):
    """
    The class defines functions with many parameters.
    
    Attributes of the class:
    :param list D:
            List of parameters (distributions), interpreted here as the marginals.
    :param numpy-matrix R:
            The correlation matrix associated with the joint distribution.
    :param object std:
            A standard normal distribution
    :param numpy-matrix A:
            The Cholesky decomposition of Fictive matrix R0, 
            associated with the set of normal intermediate
            correlated distributions.        
    """
    def __init__(self, list_of_parameters=None, correlation_matrix=None):
        if list_of_parameters is None:
            raise(ValueError, 'Distributions must be given')
        else:
            self.list_of_parameters = list_of_parameters
        self.correlation_flag = 1
        if correlation_matrix is None:
            self.correlation_matrix = np.eye(len(self.list_of_parameters))
            self.correlation_flag = 0
        else:
            self.correlation_matrix = correlation_matrix
        
        # Definition of a standard parameter!
        self.std = Parameter(order=5, distribution='normal',shape_parameter_A = 0.0, shape_parameter_B = 1.0)
        #  
        #    R0 = fictive matrix of correlated normal intermediate variables
        #
        #    1) Check the type of correlated marginals
        #    2) Use Effective Quadrature for solving Legendre
        #    3) Calculate the fictive matrix
        
        if self.correlation_matrix == 1:
            inf_lim = -8.0
            sup_lim = - inf_lim
            p1 = Parameter(distribution = 'uniform', lower = inf_lim, upper = sup_lim, order = 31)
            myBasis = Basis('Tensor grid')
            Pols = Polyint([p1, p1], myBasis)
            p = Pols.quadraturePoints
            w = Pols.quadratureWeights * (sup_lim - inf_lim)**2
        
            p1 = p[:,0]
            p2 = p[:,1]
            
            R0 = np.eye((len(self.list_of_parameters)))
            for i in range(len(self.list_of_parameters)): 
                for j in range(i+1, len(self.list_of_parameters), 1):
                    if self.correlation_matrix[i,j] == 0:
                        R0[i,j] = 0.0
                    else: 
                        tp11 = -(np.array(self.list_of_parameters[i].getiCDF(self.std.getCDF(points=p1))) - self.list_of_parameters[i].mean ) / np.sqrt( self.list_of_parameters[i].variance ) 
                        tp22 = -(np.array(self.list_of_parameters[j].getiCDF(self.std.getCDF(points=p2))) -  self.list_of_parameters[j].mean)/np.sqrt( self.list_of_parameters[j].variance )
                    
                        rho_ij = self.correlation_matrix[i,j]
                        bivariateNormalPDF = (1.0 / (2.0 * np.pi * np.sqrt(1.0-rho_ij**2)) * np.exp(-1.0/(2.0*(1.0 - rho_ij**2)) * (p1**2 - 2.0 * rho_ij * p1 * p2  + p2**2 )))
                        coefficientsIntegral = np.flipud(tp11*tp22 * w)

                        def check_difference(rho_ij):
                            bivariateNormalPDF = (1.0 / (2.0 * np.pi * np.sqrt(1.0-rho_ij**2)) * np.exp(-1.0/(2.0*(1.0 - rho_ij**2)) * (p1**2 - 2.0 * rho_ij * p1 * p2  + p2**2 )))
                            diff = np.dot(coefficientsIntegral, bivariateNormalPDF) 
                            return diff - self.correlation_matrix[i,j] 
                    
                        if (self.list_of_parameters[i].name!='custom') or (self.list_of_parameters[j].name!='custom'):
                            rho = optimize.newton(check_difference, self.correlation_matrix[i,j], maxiter=50)
                        else: 
                            res = optimize.least_squares(check_difference, R[i,j], bounds=(-0.999,0.999), ftol=1.e-03) 
                            rho = res.x

                        R0[i,j] = rho
                        R0[j,i] = R0[i,j] 

            self.A = np.linalg.cholesky(R0) 
    
    def C2U(self, X):
        """  Method for mapping correlated variables to a new standard space.
             The imput matrix must have [Nxm] dimension, where m is the number
             of correlated marginals.
             
             :param numpy-matrix X: 
                    A N-by-M Matrix where input marginals are organized along columns
                    M represents the number of correlated marginals
             :return:
                    A N-by-M Matrix which contains standardized uncorrelated data.
                    The transformation of each i-th input marginal is stored along 
                    the i-th column of the output matrix.
        """
        c = X[:,0]                
        w1 = np.zeros((len(c),len(self.list_of_parameters)))
        for i in range(len(self.list_of_parameters)):
            for j in range(len(c)):
                w1[j,i] = self.list_of_parameters[i].getCDF(points=X[j,i])
                if (w1[j,i] >= 1.0):
                    w1[j,i] = 1.0 - 10**(-10)
                elif (w1[j,i] <= 0.0):
                    w1[j,i] = 0.0 + 10**(-10)
        sU = np.zeros((len(c),len(self.list_of_parameters)))
        for i in range(len(self.list_of_parameters)):
            for j in range(len(c)):
                sU[j,i] = self.std.getiCDF(w1[j,i]) 
        
        sU = np.array(sU)
        sU = sU.T
      
        xu = np.linalg.solve(self.A,sU)
        xu = np.array(xu)
        xu = xu.T

        return xu

    def U2C(self, X):
        """ Method for mapping uncorrelated variables from standard normal space
            to a new physical space in which variables are correlated.
            Input matrix must have [mxN] dimension, where m is the number of input marginals.

            :param numpy-matrix X:
                    A Matrix of M-by-N dimensions, in which uncorrelated marginals
                    are organized along rows.
            :return:
                    A N-by-M matrix in which the result of the inverse transformation
                    applied to the i-th marginal is stored along the i-th column
                    of the ouput matrix.
        """
        X = X.T

        invA = np.linalg.inv(self.A)
        Z = np.linalg.solve(invA, X)
        Z = Z.T

        xc = np.zeros((len(Z[:,0]), len(self.list_of_parameters)))
        for i in range(len(self.list_of_parameters)):
            for j in range(len(Z[:,0])): 
                xc[j,i] = self.std.getCDF(points=Z[j,i]) 
        Xc = np.zeros((len(Z[:,0]),len(self.list_of_parameters)))
        for i in range(len(self.list_of_parameters)):
            for j in range(len(Z[:,0])):
                temporary = np.matrix(xc[j,i])
                temp = self.list_of_parameters[i].getiCDF(temporary)
                
                t = temp[0]
                Xc[j,i] = t 
        return Xc
    
    def normalizedSamples(self, N=None):
        """
        Method for generating normalized samples. For a uniform distribution

        """

    def getUncorrelatedSamples(self, N=None):
        """ Method for sampling uncorrelated data: 

            :param integer N:
                    represents the number of the samples inside a range
            :return:
                    A N-by-M matrix, each i-th column contains the points
                    which belong to the i-th distribution stored into list D.
        """
        if N is not None: 
            distro = list() 
            for i in range(len(self.list_of_parameters)): 
                    distro1 = self.list_of_parameters[i].getSamples(N)
                    
                    # check dimensions ------------------#
                    distro1 = np.matrix(distro1)
                    dimension = np.shape(distro1)
                    if dimension[0] == N:
                        distro1 = distro1.T
                    #------------------------------------#
                    distro.append(distro1) 
                                                                                                                                                                                  
            distro = np.reshape(distro, (len(self.list_of_parameters),N)) 
            distro = distro.T
       
        else:
             raise(ValueError, 'One input must be given to "get Correlated Samples" method')   
        return distro
  
    def getCorrelatedSamples(self, N=None):
        """ Method for sampling correlated data:

            :param integer N:
                represents the number of the samples inside a range
                points represents the array we want to correlate.
            
            :return:
                A N-by-M matrix in which correlated samples are organized
                along columns: the result of the run of the present method
                for the i-th marginal into the input matrix is stored 
                along the i-th column of the output matrix.
        """
        if N is not None: 
  
            distro = list() 
            for i in range(len(self.list_of_parameters)): 
                    distro1 = self.std.getSamples(N)
                    
                    # check dimensions ------------------#
                    distro1 = np.matrix(distro1)
                    dimension = np.shape(distro1)
                    if dimension[0] == N:
                        distro1 = distro1.T
                    #------------------------------------#
                    distro.append(distro1) 

            distro = np.reshape(distro, (len(self.list_of_parameters),N))
            interm = np.dot(self.A, distro)
            correlated = np.zeros((len(self.list_of_parameters),N))
            for i in range(len(self.list_of_parameters)):
                for j in range(N):
                    correlated[i,j] = self.list_of_parameters[i].getiCDF(self.std.getCDF(interm[i,j]))
            correlated = correlated.T
            return correlated
        
        else:
             raise(ValueError, 'One input must be given to "get Correlated Samples" method: please choose between sampling N points or giving an array of uncorrelated data ')   

    def CorrelationMatrix(self, X):
        """ The following calculations check the correlation
            matrix of input arrays and determine the covariance 
            matrix: The input matrix mush have [Nxm] dimensions where
            m is the number of the marginals.
            
            :param X:
                Matrix of correlated data
            :param D:
                diagonal matrix which cointains the variances
            :param S:
                covariance matrix
            :return:
                A correlation matrix R           
        """
        N = len(X)
        D = np.zeros((len(self.list_of_parameters),len(self.list_of_parameters)))
        for i in range(len(self.list_of_parameters)):
            for j in range(len(self.list_of_parameters)):
                if i==j:
                    D[i,j] = np.sqrt(self.list_of_parameters[i].variance)
                else:
                    D[i,j] = 0
        diff1 = np.zeros((N, len(self.list_of_parameters))) # (x_j - mu_j)
        diff2 = np.zeros((N, len(self.list_of_parameters))) # (x_k - mu_k)
        prod_n = np.zeros(N)
        prod_square1 = np.zeros(N)
        prod_square2 = np.zeros(N)
                                                                    
        R = np.zeros((len(self.list_of_parameters),len(self.list_of_parameters)))
        for j in range(len(self.list_of_parameters)):
            for k in range(len(self.list_of_parameters)):
                if j==k:
                    R[j,k] = 1.0
                else:
                    for i in range(N):            
                        diff1[i,j] = (X[i,j] - self.list_of_parameters[j].mean)
                        diff2[i,k] = (X[i,k] - self.list_of_parameters[k].mean)
                        prod_n[i]  = 1.0*(diff1[i,j]*diff2[i,k])
                        prod_square1[i] = (diff1[i,j])**2
                        prod_square2[i] = (diff2[i,k])**2
                                                                    
                    den1   = np.sum(prod_square1)
                    den2   = np.sum(prod_square2)
                    den11  = np.sqrt(den1)
                    den22  = np.sqrt(den2)
                    R[j,k] = np.sum(prod_n)/(den11*den22)
         
        return R
