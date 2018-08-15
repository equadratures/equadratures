from unittest import TestCase
import unittest
from equadratures import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf, gamma

class Test_Nataf(TestCase):
    """ this class compares:
        - the mean and the variance using numerical and analytical methods;
        - the Nataf transformation for couples of identical distributions.
    """ 
    def blackbox(self, x):
        return x

    def mean_variance_estimation(self, D, o):
         # testing the mean and the variance
         print '---------------------------------------------------'
         for i in range(len(D)):
             print 'mean of', i, 'output', np.mean(o[:,i]), '(', D[i].name, ')'
         for i in range(len(D)):
             print 'variance of',i,'output', np.var(o[:,i]), '(', D[i].name, ')'
         print '---------------------------------------------------'

    def test_gamma(self):
        """ A gamma distribution has the support over (0, +inf)
        """       
        x = np.linspace(0.0, 20.0, 100)
        #   parameters:
        #   k > 0
        #   theta > 0
        k     = 2.0
        theta = 0.9
        # analytical mean and variance:
        a_mean     = k*theta
        a_variance = k*theta**2
        #
        print '############################################'
        print 'Test Gamma:'
        print '--------------------------------------------'
        print 'analytical mean:', a_mean
        print 'analytical variance', a_variance 

        # numerical solution using EffectiveQuadrature:
        xo = Parameter(order=5, distribution='Gamma', lower = 0.0, upper =20.0, shape_parameter_A = 2.0, shape_parameter_B = 0.9)
        myBasis = Basis('Tensor')
        myPoly = Polyint([xo], myBasis)
        myPoly.computeCoefficients(self.blackbox)
        myStats = myPoly.getStatistics()
        print 'EffectiveQuadrature mean:', myStats.mean
        print 'EffectiveQuadrature variance:', myStats.variance
        
        # numerical solution using MonteCarlo:
        N = 900000
        xi = np.random.gamma(k,theta,(N,1))
        yi = evalfunction(np.reshape(xi, (N,1)),self.blackbox)
        print 'MonteCarlo mean:', np.mean(yi)
        print 'MonteCarlo variance', np.var(yi)

        # Estimation of error
        #   mean:
        eq_m = float('%.4f' %myStats.mean)
        mc_m = float('%.4f' %np.mean(yi))

        error_mean = np.testing.assert_almost_equal(eq_m, mc_m, decimal=2, err_msg = "Difference greated than imposed tolerance for mean value")

        #   variance:
        eq_v = float('%.4f' %myStats.variance)
        mc_v = float('%.4f' %np.var(yi))

        error_var = np.testing.assert_almost_equal(eq_v, mc_v, decimal =1, err_msg = "Difference greater than imposed tolerance for variance value")

        # test of nataf transformation
        yo = Parameter(order = 5, distribution ='gamma', shape_parameter_A = 1.7, shape_parameter_B = 0.8 )

        D = list()
        D.append(xo)
        D.append(yo)
        # correlation matrix for Nataf test:
        R = np.identity(len(D))
        for i in range(len(D)):
             for j in range(len(D)):
                 if i==j:
                      continue
                 else:
                      R[i,j] = 0.60

        # instance of Nataf class:
        obj = Nataf(D,R)

        o  = obj.getCorrelatedSamples(N = 300)
        oo = obj.getUncorrelatedSamples(N = 300)

        # testing direct transformation:
        u = obj.C2U(o)
      
        # testing the inverse transformation:
        c = obj.U2C(u)

        # testing mean and variance before correlation:
        print 'before getCorrelated method:'
        self.mean_variance_estimation(D,oo)
        # testing the mean and variance after correlation:
        print 'after getCorrelated method'
        self.mean_variance_estimation(D,o)
        # testing mean and variance of direct transformation:
        print 'Standard space:'
        self.mean_variance_estimation(D,u)
        # testing mean and variance of inverse transformation:
        print 'Physical space:'
        self.mean_variance_estimation(D,c)

    def test_beta(self):
        """ A beta distribution has the support over [0, 1] or (0,1)
        """  
        x = np.linspace(0.0, 1.0, 100)
        #   parameters:
        #   a > 0
        #   b > 0
        shape_A  = 2.0
        shape_B  = 3.0
        # analytical mean and variance:
        a_mean     = shape_A/(shape_A + shape_B)
        a_variance = (shape_A*shape_B)/((shape_A+shape_B+1)*(shape_A+shape_B)**2)

        print '############################################'
        print 'Test Beta:'
        print '--------------------------------------------'
        print 'analytical mean:', a_mean
        print 'analytical variance', a_variance 
                                                                                                                                                     
        # numerical solution using EffectiveQuadrature:
        xo = Parameter(order=5, distribution='Beta', lower = 0.0, upper =1.0, shape_parameter_A = shape_A, shape_parameter_B = shape_B)
        myBasis = Basis('Tensor')
        myPoly = Polyint([xo], myBasis)
        myPoly.computeCoefficients(self.blackbox)
        myStats = myPoly.getStatistics()
        print 'EffectiveQuadrature mean:', myStats.mean
        print 'EffectiveQuadrature variance:', myStats.variance
        
        # numerical solution using MonteCarlo:
        N = 900000
        xi = np.random.beta(shape_A,shape_B,(N,1))
        yi = evalfunction(np.reshape(xi, (N,1)), self.blackbox)
        print 'MonteCarlo mean:', np.mean(yi)
        print 'MonteCarlo variance', np.var(yi)
                                                                                                                                                     
        # Estimation of error
        #   mean:
        eq_m = float('%.4f' %myStats.mean)
        mc_m = float('%.4f' %np.mean(yi))
                                                                                                                                                     
        error_mean = np.testing.assert_almost_equal(eq_m, mc_m, decimal=2, err_msg = "Difference greated than imposed tolerance for mean value")
                                                                                                                                                     
        #   variance:
        eq_v = float('%.4f' %myStats.variance)
        mc_v = float('%.4f' %np.var(yi))
                                                                                                                                                     
        error_var = np.testing.assert_almost_equal(eq_v, mc_v, decimal =1, err_msg = "Difference greater than imposed tolerance for variance value")
                                                                                                                                                     
        # test of nataf transformation
        yo = Parameter(order = 5, distribution ='Beta', shape_parameter_A = shape_A, shape_parameter_B = shape_B )
                                                                                                                                                     
        D = list()
        D.append(xo)
        D.append(yo)
        # correlation matrix for Nataf test:
        R = np.identity(len(D))
        for i in range(len(D)):
             for j in range(len(D)):
                 if i==j:
                      continue
                 else:
                      R[i,j] = 0.60
                                                                                                                                                     
        # instance of Nataf class:
        obj = Nataf(D,R)
                                                                                                                                                     
        o  = obj.getCorrelatedSamples(N = 300)
        oo = obj.getUncorrelatedSamples(N = 300) 

        # testing direct transformation:
        u = obj.C2U(o)
      
        # testing the inverse transformation:
        c = obj.U2C(u)

        # testing mean and variance before correlation:
        print 'before getCorrelated method:'
        self.mean_variance_estimation(D,oo)
        # testing the mean and variance after correlation:
        print 'after getCorrelated method'
        self.mean_variance_estimation(D,o)
        # testing mean and variance of direct transformation:
        print 'Standard space:'
        self.mean_variance_estimation(D,u)
        # testing mean and variance of inverse transformation:
        print 'Physical space:'
        self.mean_variance_estimation(D,c)


    def test_weibull(self):
        """ A weibull distribution has the support over [0, +inf)
        """  
        #   parameters:
        #   lambda  (0, +inf) -scale factor
        #   k (0, +inf) -shape factor
        lambdaa = 1.0
        k       = 0.5
        # analytical mean and variance:
        a_mean     = lambdaa*gamma(1. +1./k)
        a_variance = lambdaa**2 * ((gamma(1. + 2. /k))-(gamma(1.+1./k))**2)
        print '############################################'
        print 'Test Weibull:'
        print '--------------------------------------------'
        print 'analytical mean:', a_mean
        print 'analytical variance', a_variance 
                                                                                                                                                     
        # numerical solution using EffectiveQuadrature:
        xo = Parameter(order=15, distribution='Weibull', shape_parameter_A = lambdaa, shape_parameter_B = k)
        myBasis = Basis('Tensor')
        myPoly = Polyint([xo], myBasis)
        myPoly.computeCoefficients(self.blackbox)
        myStats = myPoly.getStatistics()
        print 'EffectiveQuadrature mean:', myStats.mean
        print 'EffectiveQuadrature variance:', myStats.variance
        
        # numerical solution using MonteCarlo:
        N = 900000
        xi = np.random.weibull(k,(N,1))
        yi = evalfunction(np.reshape(xi, (N,1)), self.blackbox)
        print 'MonteCarlo mean:', np.mean(yi)
        print 'MonteCarlo variance', np.var(yi)
                                                                                                                                                     
        # Estimation of error
        #   mean:
        eq_m = float('%.4f' %myStats.mean)
        mc_m = float('%.4f' %np.mean(yi))
                                                                                                                                                     
        #error_mean = np.testing.assert_almost_equal(eq_m, mc_m, decimal=2, err_msg = "Difference greated than imposed tolerance for mean value")
                                                                                                                                                     
        #   variance:
        eq_v = float('%.4f' %myStats.variance)
        mc_v = float('%.4f' %np.var(yi))
                                                                                                                                                     
        #error_var = np.testing.assert_almost_equal(eq_v, mc_v, decimal =1, err_msg = "Difference greater than imposed tolerance for variance value")
                                                                                                                                                     
        # test of nataf transformation
        yo = Parameter(order = 5, distribution ='Weibull', shape_parameter_A = 0.8, shape_parameter_B = 0.9 )
                                                                                                                                                     
        D = list()
        D.append(xo)
        D.append(yo)
        # correlation matrix for Nataf test:
        R = np.identity(len(D))
        for i in range(len(D)):
             for j in range(len(D)):
                 if i==j:
                      continue
                 else:
                      R[i,j] = 0.60
                                                                                                                                                     
        # instance of Nataf class:
        obj = Nataf(D,R)
                                                                                                                                                     
        o  = obj.getCorrelatedSamples(N = 300)
        oo = obj.getUncorrelatedSamples(N = 300 )
                                                                                                                                                     
        # testing direct transformation:
        u = obj.C2U(o)
        
        # testing the inverse transformation:
        c = obj.U2C(u)

        # testing mean and variance before correlation:
        print 'before getCorrelated method:'
        self.mean_variance_estimation(D,oo)
        # testing the mean and variance after correlation:
        print 'after getCorrelated method'
        self.mean_variance_estimation(D,o)
        # testing mean and variance of direct transformation:
        print 'Standard space:'
        self.mean_variance_estimation(D,u)
        # testing mean and variance of inverse transformation:
        print 'Physical space:'
        self.mean_variance_estimation(D,c)

    def test_truncated_gauss(self):
        """ A truncated-gaussian distribution has the support over [a, b]
        """         
        x = np.linspace(10**(-10), 50.0, 100)
        #   parameters:
        #   a  real number: lower
        #   b  real number: upper
        #   sigma': deviation of parent gaussian distribution
        #   mean' : mean of the parent gaussian distribution
        mu    = 100.0
        sigma = 25.0
        a     = 50.0
        b     = 150.0
        alpha = (a-mu)/sigma
        beta  = (b-mu)/sigma 
        std = Parameter(order=5, distribution='gaussian', shape_parameter_A = 0.0, shape_parameter_B = 1.0)
        num = std.getPDF(points=beta)-std.getPDF(points=alpha)
        den = std.getCDF(points = beta)- std.getCDF(points=alpha)
        mean = mu - sigma*(num/den)
        num_i = beta*std.getPDF(points=beta)- alpha*std.getPDF(points=alpha)
        den = std.getCDF(points=beta)-std.getCDF(points=alpha)
        num_ii= std.getPDF(points=beta)-std.getPDF(points=alpha)
        variance = sigma**2 * (1. -(num_i/den)- (num_ii/den)**2)

        # analytical mean and variance:
        a_mean     = mean
        a_variance = variance
        print '############################################'
        print 'Truncated Gaussian:'
        print '--------------------------------------------'
        print 'analytical mean:', a_mean
        print 'analytical variance', a_variance 
                                                                                                                                                     
        # numerical solution using EffectiveQuadrature:
        xo = Parameter(order=5, distribution='truncated-gaussian', lower = 50., upper =150., shape_parameter_A = 100., shape_parameter_B = 25.**2)
        myBasis = Basis('Tensor')
        myPoly = Polyint([xo], myBasis)
        myPoly.computeCoefficients(self.blackbox)
        myStats = myPoly.getStatistics()
        print 'EffectiveQuadrature mean:', myStats.mean
        print 'EffectiveQuadrature variance:', myStats.variance
        
        # numerical solution using MonteCarlo:
        #N = 900000
        #xi = np.random.(k,(N,1))
        #yi = evalfunction(np.reshape(xi, (N,1)), self.blackbox)
        #print 'MonteCarlo mean:', np.mean(yi)
        #print 'MonteCarlo variance', np.var(yi)
                                                                                                                                                     
        # Estimation of error
        #   mean:
        #eq_m = float('%.4f' %myStats.mean)
        #mc_m = float('%.4f' %np.mean(yi))
                                                                                                                                                     
        #error_mean = np.testing.assert_almost_equal(eq_m, mc_m, decimal=2, err_msg = "Difference greated than imposed tolerance for mean value")
                                                                                                                                                     
        #   variance:
        #eq_v = float('%.4f' %myStats.variance)
        #mc_v = float('%.4f' %np.var(yi))
                                                                                                                                                     
        #error_var = np.testing.assert_almost_equal(eq_v, mc_v, decimal =1, err_msg = "Difference greater than imposed tolerance for variance value")
                                                                                                                                                     
        # test of nataf transformation
        yo = Parameter(order = 5, distribution ='truncated-gaussian', shape_parameter_A = 100., shape_parameter_B = 25.**2, lower = 50., upper=150. )
                                                                                                                                                     
        D = list()
        D.append(xo)
        D.append(yo)
        # correlation matrix for Nataf test:
        R = np.identity(len(D))
        for i in range(len(D)):
             for j in range(len(D)):
                 if i==j:
                      continue
                 else:
                      R[i,j] = 0.60
                                                                                                                                                     
        # instance of Nataf class:
        obj = Nataf(D,R)
                                                                                                                                                     
        o  = obj.getCorrelatedSamples(N = 300)
        oo = obj.getUncorrelatedSamples(N = 300 )

        # testing direct transformation:
        u = obj.C2U(o)
     
        # testing the inverse transformation:
        c = obj.U2C(u)
        
        # testing mean and variance before correlation:
        print 'before getCorrelated method:'
        self.mean_variance_estimation(D,oo)
        # testing the mean and variance after correlation:
        print 'after getCorrelated method'
        self.mean_variance_estimation(D,o)
        # testing mean and variance of direct transformation:
        print 'Standard space:'
        self.mean_variance_estimation(D,u)
        # testing mean and variance of inverse transformation:
        print 'Physical space:'
        self.mean_variance_estimation(D,c)

    def test_arcine(self):
        """#An arcisne distribution has the support over [a,b]
        """  
        #   parameters:
        #   a real number
        #   b real number
        a   = 0.0
        b   = 1.0
        x = np.linspace(a, b, 100)
        # analytical mean and variance:
        a_mean     = (a+b)/2.0
        a_variance = (1.0/8.0)*(b-a)**2
        print '############################################'
        #
        print 'Test Chebyshev:'
        print '--------------------------------------------'
        print 'analytical mean:', a_mean
        print 'analytical variance', a_variance 
                                                                                                                                                     
        # numerical solution using EffectiveQuadrature:
        xo = Parameter(order=5, distribution='Chebyshev', lower = 0.010, upper =0.99)
        myBasis = Basis('Tensor')
        myPoly = Polyint([xo], myBasis)
        myPoly.computeCoefficients(self.blackbox)
        myStats = myPoly.getStatistics()
        print 'EffectiveQuadrature mean:', myStats.mean
        print 'EffectiveQuadrature variance:', myStats.variance
        
        # numerical solution using MonteCarlo:
        #N = 900000
        #xi = np.random.gamma(k,theta,(N,1))
        #yi = evalfunction(np.reshape(xi, (N,1)), self.blackbox)
        #print 'MonteCarlo mean:', np.mean(yi)
        #print 'MonteCarlo variance', np.var(yi)
                                                                                                                                                     
        # Estimation of error
        #   mean:
        #eq_m = float('%.4f' %myStats.mean)
        #mc_m = float('%.4f' %np.mean(yi))
                                                                                                                                                     
        #error_mean = np.testing.assert_almost_equal(eq_m, mc_m, decimal=2, err_msg = "Difference greated than imposed tolerance for mean value")
                                                                                                                                                     
        #   variance:
        #eq_v = float('%.4f' %myStats.variance)
        #mc_v = float('%.4f' %np.var(yi))
                                                                                                                                                     
        #error_var = np.testing.assert_almost_equal(eq_v, mc_v, decimal =1, err_msg = "Difference greater than imposed tolerance for variance value")
                                                                                                                                                     
        # test of nataf transformation
        yo = Parameter(order = 5, distribution ='Chebyshev', upper = 1.0, lower = 0.0 )
                                                                                                                                                     
        D = list()
        D.append(xo)
        D.append(yo)
        # correlation matrix for Nataf test:
        R = np.identity(len(D))
        for i in range(len(D)):
             for j in range(len(D)):
                 if i==j:
                      continue
                 else:
                      R[i,j] = 0.60
                                                                                                                                                     
        # instance of Nataf class:
        obj = Nataf(D,R)
                                                                                                                                                     
        o  = obj.getCorrelatedSamples(N = 300)
        oo = obj.getUncorrelatedSamples(N = 300 )
                                                                                                                                                     
        # testing direct transformation:
        u = obj.C2U(o)
        
        # testing the inverse transformation:
        c = obj.U2C(u)
                                                                                                                                                     
        # testing mean and variance before correlation:
        print 'before getCorrelated method:'
        self.mean_variance_estimation(D,oo)
        # testing the mean and variance after correlation:
        print 'after getCorrelated method'
        self.mean_variance_estimation(D,o)
        # testing mean and variance of direct transformation:
        print 'Standard space:'
        self.mean_variance_estimation(D,u)
        # testing mean and variance of inverse transformation:
        print 'Physical space:'
        self.mean_variance_estimation(D,c)
    
    def test_gaussian(self):                                                                                                                             
     """ A gaussian distribution has the support over (-inf, +inf)
     """  
     #   parameters:
     #   variance > 0
     #   mean: real number
     var    = 2.5
     mean   = 14.0
     
     # analytical mean and variance:
     a_mean     = mean
     a_variance = var
     print '############################################'
     #
     print 'Test Gaussian:'
     print '--------------------------------------------'
     print 'analytical mean:', a_mean
     print 'analytical variance', a_variance 
                                                                                                                                                  
     # numerical solution using EffectiveQuadrature:
     xo = Parameter(order=5, distribution='gaussian', shape_parameter_A = mean, shape_parameter_B = var)
     myBasis = Basis('Tensor')
     myPoly = Polyint([xo], myBasis)
     myPoly.computeCoefficients(self.blackbox)
     myStats = myPoly.getStatistics()
     print 'EffectiveQuadrature mean:', myStats.mean
     print 'EffectiveQuadrature variance:', myStats.variance
     
     # numerical solution using MonteCarlo:
     N = 900000
     xi = np.random.normal(mean, np.sqrt(var),(N,1))
     yi = evalfunction(np.reshape(xi, (N,1)), self.blackbox)
     print 'MonteCarlo mean:', np.mean(yi)
     print 'MonteCarlo variance', np.var(yi)
                                                                                                                                                  
     # Estimation of error
     #   mean:
     eq_m = float('%.4f' %myStats.mean)
     mc_m = float('%.4f' %np.mean(yi))
                                                                                                                                                  
     error_mean = np.testing.assert_almost_equal(eq_m, mc_m, decimal=2, err_msg = "Difference greated than imposed tolerance for mean value")
                                                                                                                                                  
     #   variance:
     eq_v = float('%.4f' %myStats.variance)
     mc_v = float('%.4f' %np.var(yi))
                                                                                                                                                  
     error_var = np.testing.assert_almost_equal(eq_v, mc_v, decimal =1, err_msg = "Difference greater than imposed tolerance for variance value")
                                                                                                                                                  
     # test of nataf transformation
     yo = Parameter(order = 5, distribution ='gaussian', shape_parameter_A = 3.0, shape_parameter_B = 4.0 )
                                                                                                                                                  
     D = list()
     D.append(xo)
     D.append(yo)
     # correlation matrix for Nataf test:
     R = np.identity(len(D))
     for i in range(len(D)):
          for j in range(len(D)):
              if i==j:
                   continue
              else:
                   R[i,j] = 0.60
                                                                                                                                                  
     # instance of Nataf class:
     obj = Nataf(D,R)
                                                                                                                                                  
     o  = obj.getCorrelatedSamples(N = 300)
     oo = obj.getUncorrelatedSamples(N = 300 )

     # testing direct transformation:
     u = obj.C2U(o)

     # testing the inverse transformation:
     c = obj.U2C(u)
    
    
     # testing mean and variance before correlation:
     print 'before getCorrelated method:'
     self.mean_variance_estimation(D,oo)
     # testing the mean and variance after correlation:
     print 'after getCorrelated method'
     self.mean_variance_estimation(D,o)
     # testing mean and variance of direct transformation:
     print 'Standard space:'
     self.mean_variance_estimation(D,u)
     # testing mean and variance of inverse transformation:
     print 'Physical space:'
     self.mean_variance_estimation(D,c)

    def test_rayleigh(self):
        """ A rayleigh distribution has the support over (-inf, +inf)
        """  
        #   parameters:
        #   scale > 0
        #   
        scale    = 2.5
        
        # analytical mean and variance:
        a_mean     = scale*np.sqrt(np.pi/2.)
        a_variance = (4. - np.pi)/2. * scale**2
        print '############################################'
        #
        print 'Test Rayleigh:'
        print '--------------------------------------------'
        print 'analytical mean:', a_mean
        print 'analytical variance', a_variance 
                                                                                                                                                     
        # numerical solution using EffectiveQuadrature:
        xo = Parameter(order=5, distribution='rayleigh', shape_parameter_A = scale)
        myBasis = Basis('Tensor')
        myPoly = Polyint([xo], myBasis)
        myPoly.computeCoefficients(self.blackbox)
        myStats = myPoly.getStatistics()
        print 'EffectiveQuadrature mean:', myStats.mean
        print 'EffectiveQuadrature variance:', myStats.variance
        
        # numerical solution using MonteCarlo:
        N = 900000
        xi = np.random.rayleigh(scale,(N,1))
        yi = evalfunction(np.reshape(xi, (N,1)),self. blackbox)
        print 'MonteCarlo mean:', np.mean(yi)
        print 'MonteCarlo variance', np.var(yi)
                                                                                                                                                     
        # Estimation of error
        #   mean:
        eq_m = float('%.4f' %myStats.mean)
        mc_m = float('%.4f' %np.mean(yi))
                                                                                                                                                     
        error_mean = np.testing.assert_almost_equal(eq_m, mc_m, decimal=2, err_msg = "Difference greated than imposed tolerance for mean value")
                                                                                                                                                     
        #   variance:
        eq_v = float('%.4f' %myStats.variance)
        mc_v = float('%.4f' %np.var(yi))
                                                                                                                                                     
        error_var = np.testing.assert_almost_equal(eq_v, mc_v, decimal =1, err_msg = "Difference greater than imposed tolerance for variance value")
                                                                                                                                                     
        # test of nataf transformation
        yo = Parameter(order = 5, distribution ='rayleigh', shape_parameter_A = 0.7 )
                                                                                                                                                     
        D = list()
        D.append(xo)
        D.append(yo)
        # correlation matrix for Nataf test:
        R = np.identity(len(D))
        for i in range(len(D)):
             for j in range(len(D)):
                 if i==j:
                      continue
                 else:
                      R[i,j] = 0.60
                                                                                                                                                     
        # instance of Nataf class:
        obj = Nataf(D,R)
                                                                                                                                                     
        o  = obj.getCorrelatedSamples(N = 300)
        oo = obj.getUncorrelatedSamples(N = 300 )

        # testing direct transformation:
        u = obj.C2U(o)
        
        # testing the inverse transformation:
        c = obj.U2C(u)

        # testing mean and variance before correlation:
        print 'before getCorrelated method:'
        self.mean_variance_estimation(D,oo)
        # testing the mean and variance after correlation:
        print 'after getCorrelated method'
        self.mean_variance_estimation(D,o)
        # testing mean and variance of direct transformation:
        print 'Standard space:'
        self.mean_variance_estimation(D,u)
        # testing mean and variance of inverse transformation:
        print 'Physical space:'
        self.mean_variance_estimation(D,c)

    def test_uniform(self):
        """ A uniform distribution has the support over (-inf, +inf)
        """  
        #   parameters:
        #   scale > 0
        #   
        lower = -1.0
        upper = 1.0
        # analytical mean and variance:
        a_mean     = 0.5*(lower + upper)
        a_variance = 1. / 12. *(upper -lower)**2
        print '############################################'
        #
        print 'Test Uniform:'
        print '--------------------------------------------'
        print 'analytical mean:', a_mean
        print 'analytical variance', a_variance 
                                                                                                                                                     
        # numerical solution using EffectiveQuadrature:
        xo = Parameter(order=5, distribution='uniform', lower = lower, upper = upper )
        myBasis = Basis('Tensor')
        myPoly = Polyint([xo], myBasis)
        myPoly.computeCoefficients(self.blackbox)
        myStats = myPoly.getStatistics()
        print 'EffectiveQuadrature mean:', myStats.mean
        print 'EffectiveQuadrature variance:', myStats.variance
        
        # numerical solution using MonteCarlo:
        N = 900000
        xi = np.random.uniform(lower, upper ,(N,1))
        yi = evalfunction(np.reshape(xi, (N,1)), self.blackbox)
        print 'MonteCarlo mean:', np.mean(yi)
        print 'MonteCarlo variance', np.var(yi)
                                                                                                                                                     
        # Estimation of error
        #   mean:
        eq_m = float('%.4f' %myStats.mean)
        mc_m = float('%.4f' %np.mean(yi))
                                                                                                                                                     
        error_mean = np.testing.assert_almost_equal(eq_m, mc_m, decimal=2, err_msg = "Difference greated than imposed tolerance for mean value")
                                                                                                                                                     
        #   variance:
        eq_v = float('%.4f' %myStats.variance)
        mc_v = float('%.4f' %np.var(yi))
                                                                                                                                                     
        error_var = np.testing.assert_almost_equal(eq_v, mc_v, decimal =1, err_msg = "Difference greater than imposed tolerance for variance value")
                                                                                                                                                     
        # test of nataf transformation
        yo = Parameter(order = 5, distribution ='uniform', lower = 0.05, upper = 0.99 )
                                                                                                                                                     
        D = list()
        D.append(xo)
        D.append(yo)
        # correlation matrix for Nataf test:
        R = np.identity(len(D))
        for i in range(len(D)):
             for j in range(len(D)):
                 if i==j:
                      continue
                 else:
                      R[i,j] = 0.60
                                                                                                                                                     
        # instance of Nataf class:
        obj = Nataf(D,R)
                                                                                                                                                     
        o  = obj.getCorrelatedSamples(N = 300)
        oo = obj.getUncorrelatedSamples(N = 300 )
                                                                                                                                                     
        # testing direct transformation:
        u = obj.C2U(o)
        
        # testing the inverse transformation:
        c = obj.U2C(u)
        
        # testing mean and variance before correlation:        
        print 'before getCorrelated method:'
        self.mean_variance_estimation(D,oo)
        # testing the mean and variance after correlation:
        print 'after getCorrelated method'
        self.mean_variance_estimation(D,o)
        # testing mean and variance of direct transformation:
        print 'Standard space:'
        self.mean_variance_estimation(D,u)
        # testing mean and variance of inverse transformation:
        print 'Physical space:'
        self.mean_variance_estimation(D,c)
        
    def test_chisquared(self):                                                                                                                              
       """ A chisquared distribution has the support over (0, +inf)
           and [0, +inf) when k ==1.
       """  
       #   parameters:
       #   k : degree of freedom
       #   
       k = 15
       
       # analytical mean and variance:
       a_mean     = k
       a_variance = 2. * k
       print '############################################'
       #
       print 'Test Uniform:'
       print '--------------------------------------------'
       print 'analytical mean:', a_mean
       print 'analytical variance', a_variance 
                                                                                                                                                    
       # numerical solution using EffectiveQuadrature:
       xo = Parameter(order=5, distribution='Chisquared', shape_parameter_A = k )
       myBasis = Basis('Tensor')
       myPoly = Polyint([xo], myBasis)
       myPoly.computeCoefficients(self.blackbox)
       myStats = myPoly.getStatistics()
       print 'EffectiveQuadrature mean:', myStats.mean
       print 'EffectiveQuadrature variance:', myStats.variance
       
       # numerical solution using MonteCarlo:
       N = 900000
       xi = np.random.chisquare(k ,(N,1))
       yi = evalfunction(np.reshape(xi, (N,1)), self.blackbox)
       print 'MonteCarlo mean:', np.mean(yi)
       print 'MonteCarlo variance', np.var(yi)
                                                                                                                                                    
       # Estimation of error
       #   mean:
       eq_m = float('%.4f' %myStats.mean)
       mc_m = float('%.4f' %np.mean(yi))
                                                                                                                                                    
       error_mean = np.testing.assert_almost_equal(eq_m, mc_m, decimal=2, err_msg = "Difference greated than imposed tolerance for mean value")
                                                                                                                                                    
       #   variance:
       eq_v = float('%.4f' %myStats.variance)
       mc_v = float('%.4f' %np.var(yi))
                                                                                                                                                    
       error_var = np.testing.assert_almost_equal(eq_v, mc_v, decimal =1, err_msg = "Difference greater than imposed tolerance for variance value")
                                                                                                                                                    
       # test of nataf transformation
       yo = Parameter(order = 5, distribution ='Chisquared', shape_parameter_A = 2.0 )
                                                                                                                                                    
       D = list()
       D.append(xo)
       D.append(yo)
       # correlation matrix for Nataf test:
       R = np.identity(len(D))
       for i in range(len(D)):
            for j in range(len(D)):
                if i==j:
                     continue
                else:
                     R[i,j] = 0.60
                                                                                                                                                    
       # instance of Nataf class:
       obj = Nataf(D,R)
                                                                                                                                                    
       o  = obj.getCorrelatedSamples(N = 300)
       oo = obj.getUncorrelatedSamples(N = 300 )
                                                                                                                                                    
       # testing direct transformation:
       u = obj.C2U(o)
      
       # testing the inverse transformation:
       c = obj.U2C(u)

       # testing mean and variance before correlation:        
       print 'before getCorrelated method:'
       self.mean_variance_estimation(D,oo)
       # testing the mean and variance after correlation:
       print 'after getCorrelated method'
       self.mean_variance_estimation(D,o)
       # testing mean and variance of direct transformation:
       print 'Standard space:'
       self.mean_variance_estimation(D,u)
       # testing mean and variance of inverse transformation:
       print 'Physical space:'
       self.mean_variance_estimation(D,c)

    def test_exponential(self):                                                                                                                              
       """ An exponential distribution has the support over [0, +inf)
       """      
       #   parameters:
       #   scale (\lambda) > 0 ; rate = 1/scale.
       #   lower = minimum value
       #   upper = maximum value
       scale = 2.
       upper = 0.0
       lower = 1.0
       
       # analytical mean and variance:
       a_mean     = 1. / scale
       a_variance = 1. / scale**2
       print '############################################'
       #
       print 'Test Exponential:'
       print '--------------------------------------------'
       print 'analytical mean:', a_mean
       print 'analytical variance', a_variance 
                                                                                                                                                    
       # numerical solution using EffectiveQuadrature:
       xo = Parameter(order=5, distribution='Exponential', shape_parameter_A = 1./scale )
       myBasis = Basis('Tensor')
       myPoly = Polyint([xo], myBasis) 
       myPoly.computeCoefficients(self.blackbox)
       myStats = myPoly.getStatistics()
       print 'EffectiveQuadrature mean:', myStats.mean
       print 'EffectiveQuadrature variance:', myStats.variance
       
       # numerical solution using MonteCarlo:
       N = 900000
       xi = np.random.exponential(1./scale ,(N,1))
       yi = evalfunction(np.reshape(xi, (N,1)), self.blackbox)
       print 'MonteCarlo mean:', np.mean(yi)
       print 'MonteCarlo variance', np.var(yi)
                                                                                                                                                    
       # Estimation of error
       #   mean:
       eq_m = float('%.4f' %myStats.mean)
       mc_m = float('%.4f' %np.mean(yi))
                                                                                                                                                    
       error_mean = np.testing.assert_almost_equal(eq_m, mc_m, decimal=2, err_msg = "Difference greated than imposed tolerance for mean value")
                                                                                                                                                    
       #   variance:
       eq_v = float('%.4f' %myStats.variance)
       mc_v = float('%.4f' %np.var(yi))
                                                                                                                                                    
       error_var = np.testing.assert_almost_equal(eq_v, mc_v, decimal =1, err_msg = "Difference greater than imposed tolerance for variance value")
                                                                                                                                                    
       # test of nataf transformation
       yo = Parameter(order = 5, distribution ='Exponential', shape_parameter_A = 0.9 )
                                                                                                                                                    
       D = list()
       D.append(xo)
       D.append(yo)
       # correlation matrix for Nataf test:
       R = np.identity(len(D))
       for i in range(len(D)):
            for j in range(len(D)):
                if i==j:
                     continue
                else:
                     R[i,j] = 0.60
                                                                                                                                                    
       # instance of Nataf class:
       obj = Nataf(D,R)
                                                                                                                                                    
       o  = obj.getCorrelatedSamples(N = 300)
       oo = obj.getUncorrelatedSamples(N = 300 )

       # testing direct transformation:
       u = obj.C2U(o)
       
       # testing the inverse transformation:
       c = obj.U2C(u)

       # testing mean and variance before correlation:        
       print 'before getCorrelated method:'
       self.mean_variance_estimation(D,oo)
       # testing the mean and variance after correlation:
       print 'after getCorrelated method'
       self.mean_variance_estimation(D,o)
       # testing mean and variance of direct transformation:
       print 'Standard space:'
       self.mean_variance_estimation(D,u)
       # testing mean and variance of inverse transformation:
       print 'Physical space:'
       self.mean_variance_estimation(D,c)
    
    def test_mixed(self):
    	""" A set of mixed distributions will be tested
    	"""
    	mean1 = 0.4
    	var1  = 1.3
    	low1  = 0.2
    	upp1  = 1.15
    	
    	mean2 = 0.7
    	var2  = 3.0
    	low2  = 0.4
    	upp2  = 0.5
    
    	D = list()
    	
    	D.append(Parameter(order=3, distribution='rayleigh', shape_parameter_A=1.0))
    	D.append(Parameter(order=3, distribution='rayleigh', shape_parameter_A=4.0))
    	D.append(Parameter(order=3, distribution='uniform', lower=0.05, upper=0.99))
    	D.append(Parameter(order=3, distribution='uniform', lower=0.5, upper=0.8))
    	D.append(Parameter(order=3, distribution='gaussian', shape_parameter_A = 1.0, shape_parameter_B=16.0))
    	D.append(Parameter(order=3, distribution='gaussian', shape_parameter_A = 3.0, shape_parameter_B = 4.0))
    	D.append(Parameter(order=3, distribution='gaussian', shape_parameter_A = 3.0, shape_parameter_B = 4.0))
    	D.append(Parameter(order=3, distribution='gaussian', shape_parameter_A = 3.0, shape_parameter_B = 4.0))
    	D.append(Parameter(order=3, distribution='gaussian', shape_parameter_A = 1.0, shape_parameter_B=16.0))
    	D.append(Parameter(order=3, distribution='gaussian', shape_parameter_A = 3.0, shape_parameter_B = 4.0))
    	D.append(Parameter(order=3, distribution='Beta', lower=0.0, upper=1.0, shape_parameter_A = 1.0, shape_parameter_B = 1.0))
    	D.append(Parameter(order=3, distribution='Beta', lower=0.0, upper=1.0, shape_parameter_A = 1.0, shape_parameter_B = 1.0))
    	D.append(Parameter(order=3, distribution='Chebyshev', upper=1.0, lower=0.0))
    	D.append(Parameter(order=3, distribution='Chebyshev', upper=0.99, lower=0.01))
    	D.append(Parameter(order=3, distribution='Chisquared', shape_parameter_A=14))
    	D.append(Parameter(order=3, distribution='Chisquared', shape_parameter_A=14))
    	D.append(Parameter(order=3, distribution='exponential', shape_parameter_A = 0.7))
    	D.append(Parameter(order=3, distribution='exponential', shape_parameter_A = 0.7))
    	D.append(Parameter(order=3, distribution='gamma', shape_parameter_A = 1.7, shape_parameter_B = 0.8))
    	D.append(Parameter(order=3, distribution='gamma', shape_parameter_A = 0.7, shape_parameter_B = 0.8))
    	D.append(Parameter(order =3, distribution='rayleigh',shape_parameter_A = 0.7))
    	D.append(Parameter(order =3, distribution='rayleigh',shape_parameter_A = 0.7))
    	D.append(Parameter(order=3, distribution='truncated-gaussian',shape_parameter_A = 100., shape_parameter_B =25.0**2, upper = 150., lower = 50.))
    	D.append(Parameter(order=3, distribution='truncated-gaussian',shape_parameter_A = 100., shape_parameter_B =25.0**2, upper = 150., lower = 50.))

    	""" A default correlation matrix is defined in the following for statement:
    	"""
    	R = np.identity(len(D))
    	for i in range(len(D)): 
       	    for j in range(len(D)):
    		if i==j:
        	   	   continue
    	        else:
        	           R[i,j] = 0.60

    	""" instance of Nataf class:
    	"""
    	obj = Nataf(D,R)

    	o = obj.getCorrelatedSamples(N=300)
    	oo = obj.getUncorrelatedSamples(N=300)

    	""" testing transformations: direct
    	"""
    	u = obj.C2U(o)

    	""" testing transformations: inverse
    	"""
    	c = obj.U2C(u)

    	""" Testing mean and variance:
    	"""
    	print 'before getCorrelated:'
    	self.mean_variance_estimation(D,oo)
    	print 'after getCorrelated:'
    	self.mean_variance_estimation(D,o)
    	print 'Standard space:'
    	self.mean_variance_estimation(D,u)
    	print 'Physical space:'
    	self.mean_variance_estimation(D,c)

                                           
if __name__== '__main__':
    unittest.main()
