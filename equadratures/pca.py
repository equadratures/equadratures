""" Class for solving the Principal Component Analysis in N-dimensional case

    Input parameters:
    D : List of Distributions: instances of Parameter class.
    R : Correlatation Matrix of Distributions which belong to D.
"""
import numpy as np
from parameter import Parameter
from numpy import linalg as LA
from numpy.linalg import inv
from polyint import Polyint
from basis import Basis

class Pca(object):
    """ The class defines a Principal Component Analysis with a Whitening Method.
        
        Attributes of the Class:
        :param list D:
            List of parameters (distributions), interpreted here as marginals
        :param numpy-matrix R:
            The correlation matrix associated with the joint distribution.
        :param object std:
            A standard normal distribution
    """
    def __init__(self, D=None, R=None):
        if D is None:
            raise(ValueError, 'Distributions must be given')
        else:
            self.D = D
        if R is None:
            raise(ValueError, 'Correlation Matrix must be specified')
        else:
            self.R = R
        self.std = Parameter(order=3, distribution='normal', shape_parameter_A = 0., shape_parameter_B = 1.)

    def C2U(self, X):
        """ Method to transform the input marginals with dimension N in a new set with N' dimension,
            where N > N'.
            The input matrix X must have [Nxm] dimension, such that m is the number of correlated
            marginals.

            :param numpy-matrix X:
                A N-by-m Matrix where the input marginals are organized along columns; m represents
                the number of correlated marginals
            :return:
                A N-by-m Matrix which contains standardized uncorrelated data, in a space of M dimensions.
                The transformation of each i-th input marginal is stored along the i-th column of the 
                output matrix.
        """
        # step 1: substract the mean of each marginal
        y = np.zeros((len(X), len(self.D)))
        for j in range(len(self.D)):
            for i in range(len(X)): 
                y[i,j] = (X[i,j] - np.mean(X[:,j]))
        #R = self.CorrelationMatrix(y)
        XT = y.T 
        
        # step 2: definition of covariance matrix S 
        #S = self.CovarianceMatrix(R)
        S = np.cov(XT)
        print 'from pca class: S covariance of transpose X-mean is', S
        
        # step 3: calculation of Eigenvalues from covariance matrix
        w, v = LA.eig(S)
        
        # step 4: defintion of the transformation through the following matrix:
        #       Rot = Rotation Matrix, contribute of eigenvectors
        #       Scl = Scaling Matrix, contribute of eigenvalues
        #       T   = Transformation matrix = Rot*Scl
        Rot = v
        Scl = np.zeros((len(w), len(w)))
        for i in range(len(w)):
            for j in range(len(w)):
                if (i==j):
                    Scl[i,j] =w[i]
                else:
                    Scl[i,j] = 0.
        Scl  = np.sqrt(Scl)
        #for i in range(len(w)):
        #    for j in range(len(w)):
        #        if i==j:
        #            Scl[i,j] = (Scl[i,j])**(-.5)
        #        else:
        #            continue
        T    = np.matmul(Rot,Scl)
        invT = inv(T)
        #transp_T = T.T
        # step 5: uncorrelated data can be obtained:
        y      = y.T
        uncorr = np.matmul(invT, y)
        uncorr = uncorr.T

        return uncorr
    
    def getUncorrelatedSamples(self, N=None):
        """ Method for sampling Uncorrelated data:

            :param integer N:
                represents the number of the samples inside a range
            :return:
                A N-by-m matrix, each i-th column contains the points
                which belong to the i-th distribution stored into D.
        """
        if N is not None:
            distro = list()
            for i in range(len(self.D)):
                distro1 = self.D[i].getSamples(N)
                # check dimensions
                distro1 = np.matrix(distro1)
                dimension = np.shape(distro1)
                if dimension[0] == N:
                    distro1 = distro1.T

                distro.append(distro1)
            distro = np.reshape(distro, (len(self.D), N))
            distro = distro.T
            return distro
        else:
                raise(ValueError, 'One input must be given to UncorrelatedSamples method')
    def getCorrelatedSamples(self, N=None):
        """ Method for sampling correlated data:
        
            :param integer N:
                represents the number of the samples inside a range
            :return:
                A N-by-m matrix in which correlated samples are organized
                along columns: the results of the run of the present method
                for the i-th marginal into the input matrix is stored 
                along the i-th column of the output matrix
        """
        if N is not None:
            distro = self.getUncorrelatedSamples(N=N)
        else:
            raise(ValueError, 'One input must be given to getCorrelatedSamples method')
        rows_of_distro = len(distro) 
        distro = distro.T
        number_of_distro = len(distro)
        # Cholesky decomposition of the Covariance Matrix
        # S = L L.T, where L.T is the transpose matrix of L
         
        S = self.CovarianceMatrix(self.R)       
        print 'from pca class: covariance matrix starting from the self.correlation matrix is:', S
        L  = np.linalg.cholesky(S)
        #for i in range(number_of_distro):
        #    for j in range(rows_of_distro):
        #        distro[i,j] = (distro[i,j] - self.D[i].mean) #/ np.sqrt(self.D[i].variance)

        XC = np.matmul(L, distro)
        XC = XC.T
        for i in range(number_of_distro):
            for j in range(rows_of_distro):
                XC[j,i] = XC[j,i] - np.mean(XC[:,i])
            print 'from pca clas: mean of ', i, 'marginal:' , np.mean(XC[:,i])
        distro = distro.T
        return XC, distro

    def U2C(self, X):
        """ Method for transforming uncorrelated data from standard normal space
            to a physical space in which variables are correlated.

            :param numpy-matrix X:
                A Matrix of N-by-M dimensions, in which uncorrelated marginals
                are organized along columns
            :return:
                A N-by-M matrix in which the results of the inverse transformation
                applied to the i-th marginal is stored along the i-th column of
                the matrix
        """ 
        #R = self.CorrelationMatrix(X)
        R = self.R
        S = self.CovarianceMatrix(R)
        print 'from pca class: covariance matrix:'
        print S
        w, v = LA.eig(S)
        Rot = v
        Scl = np.zeros((len(w), len(w)))
        for i in range(len(w)):
            for j in range(len(w)):
                if i == j :
                    Scl[i,j] = w[i]
                else :
                    Scl[i,j] = 0.
        Scl = np.sqrt(Scl)
        T   = np.matmul(Rot, Scl)
        X = X.T
        corr = np.matmul(T, X)
        corr = corr.T
        for i in range(len(X)):
            for j in range(len(self.D)):
                corr[i,j] = (corr[i,j])*np.sqrt(self.D[j].variance) + self.D[j].mean
        return corr
        """
        y = np.zeros((len(X), len(self.D)))
        for i in range(len(X)):
            for j in range(len(self.D)):
                y[i,j]  = y[i,j] *np.sqrt(self.D[j].variance) + self.D[j].mean
        S = self.CovarianceMatrix(R)
        w, v = LA.eig(S)
        Rot = v
        Scl = np.zeros((len(w), len(w)))
        for i in range(len(w)):
            for j in range(len(w)):
                if i == j :
                    Scl[i,j] = w[i]
                else :
                    Scl[i,j] = 0.
        Scl = np.sqrt(Scl)
        T   = np.matmul(Rot, Scl)
        y = y.T
        corr = np.matmul(T, y)
        corr = corr.T
        #print 'correlated from U2C:', corr
        return corr
        """
    
    def CorrelationMatrix(self, X):
        """ Method to calculate the correlation matrix of a input matrix X
            X must be a NxM matrix, where M is the number of the marginals:
            distributions have to be organized along the columns of X
        """
        diff1  = np.zeros((len(X), len(self.D)))
        diff2  = np.zeros((len(X), len(self.D)))
        prod_n = np.zeros(len(X))
        prod_square1 = np.zeros(len(X))
        prod_square2 = np.zeros(len(X))

        R = np.zeros((len(self.D), len(self.D)))
        for j in range(len(self.D)):
            for k in range(len(self.D)):
                if j==k:
                    R[j,k] = 1
                else:
                    for i in range(len(X)):
                        diff1[i,j] = (X[i,j] -self.D[j].mean)
                        diff2[i,k] = (X[i,k] -self.D[k].mean)
                        prod_n[i] = (diff1[i,j]*diff2[i,k])
                        prod_square1[i] = (diff1[i,j])**2
                        prod_square2[i] = (diff2[i,k])**2
                    den1      = np.sum(prod_square1)
                    den2      = np.sum(prod_square2)
                                                     
                    den11     = np.sqrt(den1)
                    den22     = np.sqrt(den2)
        
                    R[j,k]  = np.sum(prod_n)/(den11*den22)
        return R

    def CovarianceMatrix(self, Correlation):
        #def CovarianceMatrix(self, Correlation):
        """ Calculation of covariance matrix, starting from the correlation matrix
            and the variances of each marginal
        """
        D = np.zeros((len(self.D),len(self.D)))
        for i in range(len(self.D)):
            for j in range(len(self.D)):
                if i==j :
                    D[i,j] = np.sqrt(self.D[i].variance)
                else:
                    D[i,j] = 0
        R = Correlation
        Si = np.matmul(D, R)
        S  = np.matmul(Si, D)
        #return np.cov(X)
        return S
        
