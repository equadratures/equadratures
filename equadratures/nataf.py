""" Class for solving the Nataf transformation in N-Dimensional case, 
    for generic types of input  marginals.
    
    Input parameter: 
    D : List of Distributions: instances of Parameter class.
    R : Correlation matrix of distributions which belong to D.
"""
import numpy as np
from scipy import optimize
from parameter import Parameter
from polyint import Polyint 
from basis import Basis 
import matplotlib.pyplot as plt
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D

class Nataf(object):
    """
    The class defines a Nataf transformation.
    References for theory:
        Melchers, R., E. (Robert E.), 1945- Structural reliability analysis
        and predictions - 2nd edition - John Wiley & Sons Ltd.
        
    The input correlated marginals are mapped from their physical space to a new 
    standard normal space, in which points are uncorrelated.
    
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
    def __init__(self, D=None, R=None):
        if D is None:
            raise(ValueError, 'Distributions must be given')
        else:
            self.D = D

        if R is None:
            raise(ValueError, 'Correlation matrix must be specified')
        else:
            self.R = R
        
        self.std = Parameter(order=5, distribution='normal',shape_parameter_A = 0.0, shape_parameter_B = 1.0)
        #  
        #    R0 = fictive matrix of correlated normal intermediate variables
        #
        #    1) Check the type of correlated marginals
        #    2) Use Effective Quadrature for solving Legendre
        #    3) Calculate the fictive matrix
           
        inf_lim = -8.0
        sup_lim = - inf_lim
        p1 = Parameter(distribution = 'uniform', lower = inf_lim,upper = sup_lim, order = 9)
        myBasis = Basis('Tensor grid')
        Pols = Polyint([p1, p1], myBasis)
        p = Pols.quadraturePoints
        w = Pols.quadratureWeights * (sup_lim - inf_lim)**2 
        p1 = p[:,0]
        p2 = p[:,1]

        
        R0 = np.eye((len(self.D)))
        for i in range(len(self.D)):
            print 'marginal', i, 'is a ', self.D[i].name
            for j in range(i+1, len(self.D), 1):
                if self.R[i,j] == 0:
                    R0[i,j] = 0.0
                elif i == j:
                    R0[i,j] = 1.0 
                else:
                  tp1 = self.std.getCDF(points=p1)
                  tp2 = self.std.getCDF(points=p2)
                  tp11 = (np.array(self.D[i].getiCDF(self.std.getCDF(points=p1))) - self.D[i].mean ) / np.sqrt( self.D[i].variance )
                  tp22 = (np.array(self.D[j].getiCDF(self.std.getCDF(points=p2))) -  self.D[j].mean)/np.sqrt( self.D[j].variance )
                  rho_ij = 0.6
                  bivariateNormalPDF = (1.0 / (2.0 * np.pi * np.sqrt(1.0-rho_ij**2)) * np.exp(-1.0/(2.0*(1.0 - rho_ij**2)) * (p1**2 - 2.0 * rho_ij * p1 * p2  + p2**2 )))
                  coefficientsIntegral = tp11*tp22 * w

                  def check_difference(rho_ij):
                      bivariateNormalPDF = (1.0 / (2.0 * np.pi * np.sqrt(1.0-rho_ij**2)) * np.exp(-1.0/(2.0*(1.0 - rho_ij**2)) * (p1**2 - 2.0 * rho_ij * p1 * p2  + p2**2 )))
                      diff = np.dot(coefficientsIntegral, bivariateNormalPDF) 
                      return diff - self.R[i,j] 

                  rho = optimize.newton(check_difference, self.R[i, j],tol=1e-03)
                  R0[i,j] = rho
                  R0[j,i] = R0[i,j]                         
                  self.A = np.linalg.cholesky(R0) 
        print 'The Cholesky decomposition of fictive matrix R0 is:'
        print self.A
    
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

        w1 = np.zeros((len(c),len(self.D)))
        for i in range(len(self.D)):
            for j in range(len(c)):
                w1[j,i] = self.D[i].getCDF(points=X[j,i])
                if (w1[j,i] >= 1.0):
                    w1[j,i] = 1.0 - 10**(-10)
                elif (w1[j,i] <= 0.0):
                    w1[j,i] = 0.0 + 10**(-10)

        sU = np.zeros((len(c),len(self.D)))
        for i in range(len(self.D)):
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

        xc = np.zeros((len(Z[:,0]), len(self.D)))
        for i in range(len(self.D)):
            for j in range(len(Z[:,0])):
                #xc[j,i] = self.D[i].getCDF(points=Z[j,i])
                xc[j,i] = self.std.getCDF(points=Z[j,i]) 
        Xc = np.zeros((len(Z[:,0]),len(self.D)))
        for i in range(len(self.D)):
            for j in range(len(Z[:,0])):
                temporary = np.matrix(xc[j,i])
                temp = self.D[i].getiCDF(temporary)
                
                t = temp[0]
                Xc[j,i] = t 
        return Xc
    
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
            for i in range(len(self.D)): 
                    distro1 = self.D[i].getSamples(N)
                    
                    # check dimensions ------------------#
                    distro1 = np.matrix(distro1)
                    dimension = np.shape(distro1)
                    if dimension[0] == N:
                        distro1 = distro1.T
                    #------------------------------------#
                    distro.append(distro1) 
                                                                                                                                                                                  
            distro = np.reshape(distro, (len(self.D),N)) 
            distro = distro.T
       
        else:
             raise(ValueError, 'One input must be given to "get Correlated Samples" method')   
        return distro
  
    def getCorrelatedSamples(self, N=None, points=None):
        """ Method for sampling correlated data:

            :param integer N:
                represents the number of the samples inside a range
                points represents the array we want to correlate.
            
            :param matrix points:
                points: is the input matrix which contains che set of
                uncorrelated variables we want to correlate. In this case
                the input file must have [Nxm] dimensions, where m is the
                number of input marginals.
            :return:
                A N-by-M matrix in which correlated samples are organized
                along columns: the result of the run of the present method
                for the i-th marginal into the input matrix is stored 
                along the i-th column of the output matrix.
        """
        if N is not None: 
            distro = list() 
            for i in range(len(self.D)): 
                    distro1 = self.D[i].getSamples(N)
                    
                    # check dimensions ------------------#
                    distro1 = np.matrix(distro1)
                    dimension = np.shape(distro1)
                    if dimension[0] == N:
                        distro1 = distro1.T
                    #------------------------------------#
                    distro.append(distro1) 

            distro = np.reshape(distro, (len(self.D),N)) 
            distro = distro.T
        elif points is not None:
            distro = points
            N = len(distro[:,0])
        
        else:
             raise(ValueError, 'One input must be given to "get Correlated Samples" method: please choose between sampling N points or giving an array of uncorrelated data ')   
  
        rows_of_distro = len(distro) # marginals along columns
        distro = distro.T
        number_of_distro = len(distro)  # transpose of original matrix
       
        D = np.zeros((len(self.D),len(self.D)))
        for i in range(len(self.D)):
            for j in range(len(self.D)):
                if i==j:
                    D[i,j] = np.sqrt(self.D[i].variance)
                else:
                    D[i,j] = 0

        R = self.R
        Si     = np.matmul(D,R)
        S      = np.matmul(Si, D)
        # Once S (covariance matrix) has been calculated,
        #    the Cholesky decomposition of this later can 
        #    be carried out.
        #    S = L*L^T where L^T is the transpose matrix of L.
        #
        L    = np.linalg.cholesky(S)

        #"""  standardized distributions as inputs
        #     1) subtract the mean value of each distribution
        #     distro is now a matrix with marginals along rows.
        #     2) divide by the variance of each marginal
        #"""
        for i in range(number_of_distro):
            for j in range(rows_of_distro):
                distro[i,j] = (distro[i,j] - self.D[i].mean)/np.sqrt(self.D[i].variance)

        XC   = np.matmul(L, distro)
        XC   = XC.T
        for i in range(number_of_distro):
            for j in range(rows_of_distro):
                XC[j,i] = XC[j,i] + self.D[i].mean
        #print XC
        #""" The results will be stored in the following lines into 
        #    two different tuples: the element 0 contains the 
        #    original coordinates that have been given as inputs;
        #    the element 1 contains the results of the running
        #    of the present method.
        #"""
        distro = distro.T
        for i in range(number_of_distro):
            for j in range(rows_of_distro):
                distro[j,i] = (distro[j,i])*np.sqrt(self.D[i].variance) + self.D[i].mean

        return distro, XC          
      
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
        D = np.zeros((len(self.D),len(self.D)))
        for i in range(len(self.D)):
            for j in range(len(self.D)):
                if i==j:
                    D[i,j] = np.sqrt(self.D[i].variance)
                else:
                    D[i,j] = 0
        diff1 = np.zeros((N, len(self.D))) # (x_j - mu_j)
        diff2 = np.zeros((N, len(self.D))) # (x_k - mu_k)
        prod_n = np.zeros(N)
        prod_square1 = np.zeros(N)
        prod_square2 = np.zeros(N)
                                                                    
        R = np.zeros((len(self.D),len(self.D)))
        for j in range(len(self.D)):
            for k in range(len(self.D)):
                if j==k:
                    R[j,k] = 1.0
                else:
                    for i in range(N):            
                        diff1[i,j] = (distro[i,j] - self.D[j].mean)
                        diff2[i,k] = (distro[i,k] - self.D[k].mean)
                        prod_n[i]  = -1.0*(diff1[i,j]*diff2[i,k])
                        prod_square1[i] = (diff1[i,j])**2
                        prod_square2[i] = (diff2[i,k])**2
                                                                    
                    den1   = np.sum(prod_square1)
                    den2   = np.sum(prod_square2)
                    den11  = np.sqrt(den1)
                    den22  = np.sqrt(den2)
                    R[j,k] = np.sum(prod_n)/(den11*den22)
        
        #print R
        return R
