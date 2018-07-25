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

class Nataf(object):
    """
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
        
        self.std = Parameter(order=5, distribution='normal',shape_parameter_A = 0.0, shape_parameter_B = 1.0, lower=-90, upper = 90)
        """   
            R0 = fictive matrix of correlated normal intermediate variables
        """
        """ 1) Check the type of correlated marginals
        """    
        # Quadrature rule!
        p1 = Parameter(distribution='uniform',lower=-1.,upper =1., order=5)
        myBasis = Basis('Tensor grid')
        Pols = Polyint([p1, p1], myBasis)
        p = Pols.quadraturePoints
        w = Pols.quadratureWeights
                

        R0 = np.zeros((len(self.D),len(self.D)))
        for i in range(len(self.D)):
            j = i+1
            for j in range(len(self.D)):
                if self.R[i,j] == 0:
                    R0[i,j] = 0
                elif i == j:
                    R0[i,j] = 1             
                elif (self.D[i].name == 'normal') or (self.D[i].name == 'gaussian') and (self.D[j].name == 'normal') or (self.D[j].name == 'gaussian'):
                    R0[i,j] = self.R[i,j]
                else:
                  """ 1.a) Method for calculating the fictive matrix of intermediate normal points.
                  - Legendre method will be used for calculating weights and points of double integral.
                  """
                  
                  inf_lim = -10
                  sup_lim = 10
                  ampl    = sup_lim - inf_lim
                  """ The following lines solve Legendre with EQ tools
                  """
                  
                  p = -(0.5*(p+1)*ampl + inf_lim)
                  w = w*(0.5* ampl)
                  N = len(p)

                  test_p1 = np.tile(p, [N,1])
                  test_p1 = test_p1.flatten(order='F')
                  test_p2 = np.tile(p,N)

                  matrix1 = np.tile(w,N)
                  matrix1 = np.reshape(matrix1, [N,N])
                  matrix2 = np.transpose(matrix1)

                  surface = matrix1*matrix2
                  surface = surface.flatten()

                  """ General equation of off -diagonal element of fictive matrix
                  """
                  tp1 = self.std.getCDF(points=test_p2)
                  tp1 = np.array(tp1)
                  for k in range(len(tp1)):
                    if tp1[k] == 1.0:
                        tp1[k] = 1.0 - 10**(-10)
                    elif tp1[k] == 0.0:
                        tp1[k] = 0.0 + 10**(-10)
                  tp11 = ((self.D[j].getiCDF(tp1))-self.D[j].mean)/self.D[j].variance
                  tp2 = self.std.getCDF(points=test_p1)
                  for k in range(len(tp2)):
                    if tp2[k] == 1.0:
                        tp2[k] = 1.0 - 10.**(-10)
                    elif tp2[k] == 0.0:
                        tp2[k] = 0.0 + 10**(-10)
                  tp22 = ((self.D[i].getiCDF(tp2))-self.D[i].mean)/self.D[i].variance

                  t = tp11*tp22*surface
                  def check_difference(rho_ij):
                      if rho_ij >= 1.0:
                            rho_ij = 1.0-10**(-10)
                      elif rho_ij <= -1.0:
                            rho_ij = -1.0 +10**(-10)
                      den = 2.0*np.pi*np.sqrt(1.0 - rho_ij**2)
                      #print 'rho_ij:' , rho_ij
                      esp = (-0.5/(1- rho_ij**2))*(tp1**2-2.0*tp1*tp2+tp2**2)
                      diff = t*(1.0/den)*np.exp(esp) 
                      difR = np.sum(diff) - self.R[i,j]
                      return difR

                  rho = optimize.newton(check_difference, 0.5, tol=3.0e-05)
                  R0[i,j] = rho
                  #print 'R0[i,j]', R0[i,j]
        self.A = np.linalg.cholesky(R0)
    
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
                #Xc[j,i] = self.std.getiCDF(xc[j,i])
                Xc[j,i] = self.D[i].getiCDF(xc[j,i])
        #print Xc
        return Xc
    
    def getUncorrelatedSamples(self, N=None):
        """ Method for sampling uncorrelated data: 
            N represents the number of the samples inside a range
            points represents the array we want to uncorrelate.
        """
        if N is not None:
            distro = np.zeros((N, len(self.D)))
            #print 'first distro, initialization:'
            #print distro
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
            distro = np.zeros((N, len(self.D)))
            for i in range(len(self.D)):
                for j in range(N):
                    distro1 = self.D[i].getSamples(N)
                    distro[j,i] = distro1[j]
                print 'Distribution number:',i,'is a', self.D[i].name
        elif points is not None:
            distro = points
            N = len(distro[:,0])
        
        else:
             raise(ValueError, 'One input must be given to "get Correlated Samples" method: please choose between sampling N points or giving an array of uncorrelated data ')   
        
        distro = distro.T
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
        XC   = np.matmul(L, distro)
        XC   = XC.T
        #print XC
        """ The results will be stored in the following lines into 
            two different tuples: the element 0 contains the 
            original coordinates that have been given as inputs;
            the element 1 contains the results of the running
            of the present method.
        """
        distro = distro.T
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
