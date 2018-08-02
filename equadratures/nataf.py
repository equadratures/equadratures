""" Class for solving the Nataf transformation in ND case, for generic
    marginals.
    input parameter: 
    D : Distributions: they are instances of Parameter class.
    R : Correlation matrix
"""
import numpy as np
from scipy import optimize
from parameter import Parameter
from polyint import Polyint 
from basis import Basis 
import matplotlib.pyplot as plt
from scipy import stats

class Nataf(object):
    """
    The class defines a Nataf transformation. 
    :param list D:
		A list of parameters (distributions), interpreted here as the marginals.
	:param numpy-matrix R:
		The correlation matrix associated with the joint distribution.
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
        p1 = Parameter(distribution = 'uniform', lower = inf_lim,upper = sup_lim, order = 10)
        myBasis = Basis('Tensor grid')
        Pols = Polyint([p1, p1], myBasis)
        #p = Pols.quadraturePoints
        #w = Pols.quadratureWeights

        n = 10
        zmax = 8
        zmin = -zmax
        points, weights = np.polynomial.legendre.leggauss(n)
        points = - (0.5 * (points + 1) * (zmax - zmin) + zmin)
        weights = weights * (0.5 * (zmax - zmin))

        xi = np.tile(points, [n, 1])
        p1 = xi.flatten(order='F')
        p2 = np.tile(points, n)

        first = np.tile(weights, n)
        first = np.reshape(first, [n, n])
        second = np.transpose(first)

        weights2d = first * second
        w2d = weights2d.flatten()

        R0 = np.eye((len(self.D)))
        for i in range(len(self.D)):
            print 'marginal', i, 'is a ', self.D[i].name
            for j in range(i+1, len(self.D), 1):
                if self.R[i,j] == 0:
                    R0[i,j] = 0.0
                elif i == j:
                    R0[i,j] = 1.0 
                else:
                  #p1  = p[:,0]
                  #p2  = p[:,1]
                  tp1 = self.std.getCDF(points=p1)
                  tp2 = self.std.getCDF(points=p2)
                  tp11 = (np.array(self.D[i].getiCDF(stats.norm.cdf(p1))) - self.D[i].mean ) / np.sqrt( self.D[i].variance )
                  tp22 = (np.array(self.D[j].getiCDF(stats.norm.cdf(p2))) -  self.D[j].mean)/np.sqrt( self.D[j].variance )
                  rho_ij = 0.6
                  bivariateNormalPDF = (1.0 / (2.0 * np.pi * np.sqrt(1.0-rho_ij**2)) * np.exp(-1.0/(2.0*(1.0 - rho_ij**2)) * (p1**2 - 2.0 * rho_ij * p1 * p2  + p2**2 )))
                  coefficientsIntegral = tp11*tp22 * w2d

                  fig = plt.figure()
                  plt.plot(coefficientsIntegral * bivariateNormalPDF, '.')
                  plt.show()

                  def check_difference(rho_ij):
                      bivariateNormalPDF = (1.0 / (2.0 * np.pi * np.sqrt(1.0-rho_ij**2)) * np.exp(-1.0/(2.0*(1.0 - rho_ij**2)) * (p1**2 - 2.0 * rho_ij * p1 * p2  + p2**2 )))
                      diff = np.dot(coefficientsIntegral, bivariateNormalPDF) 
                      return diff - self.R[i,j] 
                
                  #hyp_1 = self.R[i,j]
                  #rho = optimize.newton(check_difference, self.R[i, j])
                  x0, r = optimize.brentq(f=check_difference, a=-1 + np.finfo(float).eps, b=1 - np.finfo(float).eps, full_output=True, disp=True, maxiter=300)
                  print ' got here!'
                  print r 
                  #rho = optimize.fsolve(func=check_difference, x0=self.R[i, j], full_output=True)
                  if r.converged == 1: 
                    R0[i,j] = x0
                    R0[j,i] = R0[i,j]                         
                    self.A = np.linalg.cholesky(R0) 

        print self.A
    
    def C2U(self, X):
        """  Method for mapping correlated variables to a new standard space
             The imput matrix must have [Nxm] dimension, where m is the number of correlated marginals
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
        #print 'w1:'
        #print w1
        sU = np.zeros((len(c),len(self.D)))
        for i in range(len(self.D)):
            for j in range(len(c)):
                sU[j,i] = self.std.getiCDF(w1[j,i])
        #print 'Su:'
        #print sU
        sU = np.array(sU)
        sU = sU.T

        xu = np.linalg.solve(self.A,sU)
        xu = np.array(xu)
        xu = xu.T
        #print 'xu:'
        #print xu
        return xu

    def U2C(self, X):
        """ Methof for mappint uncorrelated variables to a new physical space in which variables are correlated
            Input matrix must have [mxN] dimension, where m is the number of input marginals.
        """
        X = X.T
        #print X
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
                #temp = np.reshape(temp, (1,1))
                t = temp[0]
                Xc[j,i] = t
                #Xc[j,i] = self.D[i].getiCDF(xc[j,i])
        #print Xc
        return Xc
    
    def getUncorrelatedSamples(self, N=None):
        """ Method for sampling uncorrelated data: 
            N represents the number of the samples inside a range
            points represents the array we want to uncorrelate.
        """
        if N is not None:
            distro = np.zeros((N, len(self.D))) 
            for i in range(len(self.D)):
                for j in range(N):
                    distro1 = self.D[i].getSamples(N)
                    distro[j,i] = distro1[j]
                print 'Distribution number:',i,'is a', self.D[i].name 
         
            return distro
         
        else:
            raise(ValueError, 'One input must be given to "get Uncorrelated Samples" method: please digit the uncorrelated variables number (N)')
    
    def getCorrelatedSamples(self, N=None, points=None):
        """ Method for sampling correlated data:

            N:  represents the number of the samples inside a range
            points represents the array we want to correlate.
            
            points: is the input matrix which contains che set of
            uncorrelated variables we want to correlate. In this case
            the input file must have [Nxm] dimensions, where m is the
            number of input marginals.
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
                    print 'Distribution number:',i,'is a', self.D[i].name

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
        """ Once S (covariance matrix) has been calculated,
            the Cholesky decomposition of this later can 
            be carried out.
            S = L*L^T where L^T is the transpose matrix of L.
        """
        L    = np.linalg.cholesky(S)

        """  standardized distributions as inputs
             1) subtract the mean value of each distribution
             distro is now a matrix with marginals along rows.
             2) divide by the variance of each marginal
        """
        for i in range(number_of_distro):
            for j in range(rows_of_distro):
                distro[i,j] = (distro[i,j] - self.D[i].mean)/np.sqrt(self.D[i].variance)

        XC   = np.matmul(L, distro)
        XC   = XC.T
        for i in range(number_of_distro):
            for j in range(rows_of_distro):
                XC[j,i] = XC[j,i] + self.D[i].mean
        #print XC
        """ The results will be stored in the following lines into 
            two different tuples: the element 0 contains the 
            original coordinates that have been given as inputs;
            the element 1 contains the results of the running
            of the present method.
        """
        distro = distro.T
        for i in range(number_of_distro):
            for j in range(rows_of_distro):
                distro[j,i] = (distro[j,i])*np.sqrt(self.D[i].variance) + self.D[i].mean

        return distro, XC          
      
    def CorrelationMatrix(self, X):
        """ The following calculations check the correlation
            matrix of input arrays and determine the covariance 
            matrix:
            D = diagonal matrix which cointains the variances
            R = actual correlation matrix of input
            S = covariance matrix

            The input matrix mush have [Nxm] dimensions where
            m is the number of the marginals.
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
