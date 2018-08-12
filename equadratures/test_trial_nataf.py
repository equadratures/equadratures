from unittest import TestCase
import unittest
from equadratures import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf, gamma

class Test_Nataf(TestCase):
    """ this class compares the mean and the variance
        using numerical stretegies (EffetiveQuadrature
        and MonteCarlo) and the corresponding analycital
        values.
        The Nataf transformation will be tested for 
        couples of identical type of distributions.
    """ 

    def test_gamma(self):
        """ A gamma distribution has the support over (0, +inf)
        """  
        def blackbox(x):
            return x 
        
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
        print 'Test Gamma:'
        print '--------------------------------------------'
        print 'analytical mean:', a_mean
        print 'analytical variance', a_variance 

        # numerical solution using EffectiveQuadrature:
        xo = Parameter(order=5, distribution='Gamma', lower = 0.0, upper =20.0, shape_parameter_A = 2.0, shape_parameter_B = 0.9)
        myBasis = Basis('Tensor')
        myPoly = Polyint([xo], myBasis)
        myPoly.computeCoefficients(blackbox)
        myStats = myPoly.getStatistics()
        print 'EffectiveQuadrature mean:', myStats.mean
        print 'EffectiveQuadrature variance:', myStats.variance
        
        # numerical solution using MonteCarlo:
        N = 900000
        xi = np.random.gamma(k,theta,(N,1))
        yi = evalfunction(np.reshape(xi, (N,1)), blackbox)
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
        # distributions will be correlated
        # with the correlation defined in 
        # method 'CorrelationMatrix':
        obj = Nataf(D,R)

        o  = obj.getCorrelatedSamples(N = 300)
        oo = obj.getUncorrelatedSamples(N = 300 )
        # correlated data:
        t   = o[:,0]
        tt  = o[:,1]
        
        # plot of the data:
        # correlated VS uncorrelated input
        plt.figure()
        plt.grid(linewidth= 0.4, color='k')
        plt.plot(t, tt, 'ro', label='new correlated')
        plt.plot(oo[:,0], oo[:,1], 'bo', label='original: uncorrelated')
        plt.legend(loc='upper left')
        plt.title('Gamma: results of getCorrelatedSamples method')
        plt.axis('equal')
        plt.show()
        # check the mean and the variance after correlation:
        print '____ test the mean and the variance after getCorrelated____'
        print 'mean of uncorrelated inputs:', obj.D[0].mean, obj.D[1].mean
        print 'mean of correlated outputs:', np.mean(t), np.mean(tt)
        print 'variance of uncorrelated inputs:', obj.D[0].variance, obj.D[1].variance
        print 'variance of correlated outputs:', np.var(t), np.var(tt)

        # testing direct transformation:
        u = obj.C2U(o)
        plt.figure()
        plt.grid(linewidth=0.4, color='k')
        plt.plot(u[:,0], u[:,1], 'ro', label='Uncorrelated outputs')
        plt.plot(t ,tt, 'bo', label='Correlated inputs')
        plt.legend(loc='upper left')
        plt.title('Nataf transformation for Gamma distribution')
        plt.axis('equal')
        plt.show()
        
        # testing the inverse transformation:
        c = obj.U2C(u)
        plt.figure()
        plt.grid(linewidth=0.4, color='k')
        plt.plot(c[:,0], c[:,1], 'ro', label='Correlated output')
        plt.plot(u[:,0], u[:,1], 'bo', label='Uncorrelated input')
        plt.legend(loc='upper left')
        plt.title('Nataf inverse transformation for Gamma distribution')
        plt.axis('equal')
        plt.show()

        # check the uncorrelated input and the uncorrelated output
        plt.figure()
        plt.grid(linewidth=0.4, color='k')
        plt.plot(c[:,0], c[:,1], 'ro', label='uncorrelated out')
        plt.plot(t, tt, 'bx', label='uncorrelated input')
        plt.title('Gamma: Comparison between input of direct and output of inverse')
        plt.axis('equal')
        plt.show()

    def test_beta(self):
        """ A beta distribution has the support over [0, 1] or (0,1)
        """  
        def blackbox(x):
            return x 
        
        x = np.linspace(0.0, 1.0, 100)
        #   parameters:
        #   a > 0
        #   b > 0
        shape_A  = 2.0
        shape_B  = 3.0
        # analytical mean and variance:
        a_mean     = shape_A/(shape_A + shape_B)
        a_variance = (shape_A*shape_B)/((shape_A+shape_B+1)*(shape_A+shape_B)**2)

        #
        print 'Test Beta:'
        print '--------------------------------------------'
        print 'analytical mean:', a_mean
        print 'analytical variance', a_variance 
                                                                                                                                                     
        # numerical solution using EffectiveQuadrature:
        xo = Parameter(order=5, distribution='Beta', lower = 0.0, upper =1.0, shape_parameter_A = shape_A, shape_parameter_B = shape_B)
        myBasis = Basis('Tensor')
        myPoly = Polyint([xo], myBasis)
        myPoly.computeCoefficients(blackbox)
        myStats = myPoly.getStatistics()
        print 'EffectiveQuadrature mean:', myStats.mean
        print 'EffectiveQuadrature variance:', myStats.variance
        
        # numerical solution using MonteCarlo:
        N = 900000
        xi = np.random.beta(shape_A,shape_B,(N,1))
        yi = evalfunction(np.reshape(xi, (N,1)), blackbox)
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
        # distributions will be correlated
        # with the correlation defined in 
        # method 'CorrelationMatrix':
        obj = Nataf(D,R)
                                                                                                                                                     
        o  = obj.getCorrelatedSamples(N = 300)
        oo = obj.getUncorrelatedSamples(N = 300 )
        # correlated data:
        t   = o[:,0]
        tt  = o[:,1]
        
        # plot of the data:
        # correlated VS uncorrelated input
        
        plt.figure()
        plt.grid(linewidth= 0.4, color='k') 
        plt.plot(t, tt, 'ro', label='new: correlated') 
        plt.plot(oo[:,0], oo[:,1], 'bo', label='original: uncorrelated')
        plt.legend(loc='upper left')
        plt.title('Beta: results of getCorrelatedSamples method')
        plt.axis('equal')
        plt.show()
        
        # check the mean and the variance after correlation:
        print '____ test the mean and the variance after getCorrelated____'
        print 'mean of uncorrelated inputs:', obj.D[0].mean, obj.D[1].mean
        print 'mean of correlated outputs:', np.mean(t), np.mean(tt)
        print 'variance of uncorrelated inputs:', obj.D[0].variance, obj.D[1].variance
        print 'variance of correlated outputs:', np.var(t), np.var(tt)
                                                                                                                                                     
        # testing direct transformation:
        u = obj.C2U(o)
        plt.figure()
        plt.grid(linewidth=0.4, color='k')
        plt.plot(u[:,0], u[:,1], 'ro', label='Uncorrelated outputs')
        plt.plot(t ,tt, 'bo', label='Correlated inputs')
        plt.legend(loc='upper left')
        plt.title('Nataf transformation for Beta distribution')
        plt.axis('equal')
        plt.show()
        
        # testing the inverse transformation:
        c = obj.U2C(u)
        plt.figure()
        plt.grid(linewidth=0.4, color='k')
        plt.plot(c[:,0], c[:,1], 'ro', label='Correlated output')
        plt.plot(u[:,0], u[:,1], 'bo', label='Uncorrelated input')
        plt.legend(loc='upper left')
        plt.title('Nataf inverse transformation for Beta distribution')
        plt.axis('equal')
        plt.show()
                                                                                                                                                     
        # check the uncorrelated input and the uncorrelated output
        plt.figure()
        plt.grid(linewidth=0.4, color='k')
        plt.plot(c[:,0], c[:,1], 'ro', label='uncorrelated out')
        plt.plot(t, tt, 'bx', label='uncorrelated input')
        plt.title('Beta: Comparison between input of direct and output of inverse')
        plt.axis('equal')
        plt.show()

    def test_weibull(self):
        """ A weibull distribution has the support over [0, +inf)
        """  
        def blackbox(x):
            return x 
        
        #   parameters:
        #   lambda  (0, +inf) -scale factor
        #   k (0, +inf) -shape factor
        lambdaa = 1.0
        k       = 0.5
        # analytical mean and variance:
        a_mean     = lambdaa*gamma(1. +1./k)
        a_variance = lambdaa**2 * ((gamma(1. + 2. /k))-(gamma(1.+1./k))**2)
        #
        print 'Test Weibull:'
        print '--------------------------------------------'
        print 'analytical mean:', a_mean
        print 'analytical variance', a_variance 
                                                                                                                                                     
        # numerical solution using EffectiveQuadrature:
        xo = Parameter(order=15, distribution='Weibull', shape_parameter_A = lambdaa, shape_parameter_B = k)
        myBasis = Basis('Tensor')
        myPoly = Polyint([xo], myBasis)
        myPoly.computeCoefficients(blackbox)
        myStats = myPoly.getStatistics()
        print 'EffectiveQuadrature mean:', myStats.mean
        print 'EffectiveQuadrature variance:', myStats.variance
        
        # numerical solution using MonteCarlo:
        N = 900000
        xi = np.random.weibull(k,(N,1))
        yi = evalfunction(np.reshape(xi, (N,1)), blackbox)
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
        # distributions will be correlated
        # with the correlation defined in 
        # method 'CorrelationMatrix':
        obj = Nataf(D,R)
                                                                                                                                                     
        o  = obj.getCorrelatedSamples(N = 300)
        oo = obj.getUncorrelatedSamples(N = 300 )
        # correlated data:
        t   = o[:,0]
        tt  = o[:,1]
        
        # plot of the data:
        # correlated VS uncorrelated input
        plt.figure()
        plt.grid(linewidth= 0.4, color='k')
        plt.plot(t, tt, 'ro', label='new correlated')
        plt.plot(oo[:,0], oo[:,1], 'bo', label='original: uncorrelated')
        plt.legend(loc='upper left')
        plt.title('Weibull: results of getCorrelatedSamples method')
        plt.axis('equal')
        plt.show()
        
        # check the mean and the variance after correlation:
        print '____ test the mean and the variance after getCorrelated____'
        print 'mean of uncorrelated inputs:', obj.D[0].mean, obj.D[1].mean
        print 'mean of correlated outputs:', np.mean(t), np.mean(tt)
        print 'variance of uncorrelated inputs:', obj.D[0].variance, obj.D[1].variance
        print 'variance of correlated outputs:', np.var(t), np.var(tt)
                                                                                                                                                     
        # testing direct transformation:
        u = obj.C2U(o)
        plt.figure()
        plt.grid(linewidth=0.4, color='k')
        plt.plot(u[:,0], u[:,1], 'ro', label='Uncorrelated outputs')
        plt.plot(t ,tt, 'bo', label='Correlated inputs')
        plt.legend(loc='upper left')
        plt.title('Nataf transformation for Weibull distribution')
        plt.axis('equal')
        plt.show()
        
        # testing the inverse transformation:
        c = obj.U2C(u)
        plt.figure()
        plt.grid(linewidth=0.4, color='k')
        plt.plot(c[:,0], c[:,1], 'ro', label='Correlated output')
        plt.plot(u[:,0], u[:,1], 'bo', label='Uncorrelated input')
        plt.legend(loc='upper left')
        plt.title('Nataf inverse transformation for Weibull distribution')
        plt.axis('equal')
        plt.show()
                                                                                                                                                     
        # check the uncorrelated input and the uncorrelated output
        plt.figure()
        plt.grid(linewidth=0.4, color='k')
        plt.plot(c[:,0], c[:,1], 'ro', label='uncorrelated out')
        plt.plot(t, tt, 'bx', label='uncorrelated input')
        plt.title('Weibull: Comparison between input of direct and output of inverse')
        plt.axis('equal')
        plt.show()

    def test_truncated_gauss(self):
        """ A truncated-gaussian distribution has the support over [a, b]
        """  
        def blackbox(x):
            return x 
        
        x = np.linspace(10**(-10), 50.0, 100)
        #   parameters:
        #   a  real number: lower
        #   b  real number: upper
        #   sigma': deviation of parent gaussian distribution
        #   mean' : mean of the parent gaussian distribution
        #mu    = 100.0
        #sigma = 25.0
        #a     = 50.0
        #b     = 150.0
        #alpha = (a-mu)/sigma
        #beta  = (b-mu)/sigma 
        #std = Parameter(order=5, distribution='gaussian', shape_parameter_A = 0.0, shape_parameter_B = 1.0)
        #num = std.getPDF(points=beta)-std.getPDF(points=alpha)
        #den = std.getCDF(points = beta)- std.getCDF(points=alpha)
        #mean = mu - sigma*(num/den)
        #num_i = beta*std.getPDF(points=beta)- alpha*std.getPDF(points=alpha)
        #den = std.getCDF(points=beta)-std.getCDF(points=alpha)
        #num_ii= std.getPDF(points=beta)-std.getPDF(points=alpha)
        #variance = sigma**2 * (1. -(num_i/den)- (num_ii/den)**2)

        # analytical mean and variance:
        #a_mean     = mean
        #a_variance = variance
        #
        print 'Truncated Gaussian:'
        print '--------------------------------------------'
        #print 'analytical mean:', a_mean
        #print 'analytical variance', a_variance 
                                                                                                                                                     
        # numerical solution using EffectiveQuadrature:
        xo = Parameter(order=5, distribution='truncated-gaussian', lower = 50., upper =150., shape_parameter_A = 100., shape_parameter_B = 25.**2)
        myBasis = Basis('Tensor')
        myPoly = Polyint([xo], myBasis)
        myPoly.computeCoefficients(blackbox)
        myStats = myPoly.getStatistics()
        print 'EffectiveQuadrature mean:', myStats.mean
        print 'EffectiveQuadrature variance:', myStats.variance
        
        # numerical solution using MonteCarlo:
        #N = 900000
        #xi = np.random.(k,(N,1))
        #yi = evalfunction(np.reshape(xi, (N,1)), blackbox)
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
        #  distributions will be correlated
        # with the correlation defined in 
        # method 'CorrelationMatrix':
        obj = Nataf(D,R)
                                                                                                                                                     
        o  = obj.getCorrelatedSamples(N = 300)
        oo = obj.getUncorrelatedSamples(N = 300 )
        # correlated data:
        t   = o[:,0]
        tt  = o[:,1]
        
        # plot of the data:
        # correlated VS uncorrelated input
        plt.figure()
        plt.grid(linewidth= 0.4, color='k')
        plt.plot(t, tt, 'ro', label='new correlated')
        plt.plot(oo[:,0], oo[:,1], 'bo', label='original: uncorrelated')
        plt.legend(loc='upper left')
        plt.title('Truncated Gaussian: results of getCorrelatedSamples method')
        plt.axis('equal')
        plt.show()
        
        # check the mean and the variance after correlation:
        print '____ test the mean and the variance after getCorrelated____'
        print 'mean of uncorrelated inputs:', obj.D[0].mean, obj.D[1].mean
        print 'mean of correlated outputs:', np.mean(t), np.mean(tt)
        print 'variance of uncorrelated inputs:', obj.D[0].variance, obj.D[1].variance
        print 'variance of correlated outputs:', np.var(t), np.var(tt)
                                                                                                                                                     
        # testing direct transformation:
        u = obj.C2U(o)
        plt.figure()
        plt.grid(linewidth=0.4, color='k')
        plt.plot(u[:,0], u[:,1], 'ro', label='Uncorrelated outputs')
        plt.plot(t ,tt, 'bo', label='Correlated inputs')
        plt.legend(loc='upper left')
        plt.title('Nataf transformation for Truncated-Gaussian distribution')
        plt.axis('equal')
        plt.show()
        
        # testing the inverse transformation:
        c = obj.U2C(u)
        plt.figure()
        plt.grid(linewidth=0.4, color='k')
        plt.plot(c[:,0], c[:,1], 'ro', label='Correlated output')
        plt.plot(u[:,0], u[:,1], 'bo', label='Uncorrelated input')
        plt.legend(loc='upper left')
        plt.title('Nataf inverse transformation for Truncated-Gaussian distribution')
        plt.axis('equal')
        plt.show()
                                                                                                                                                     
        # check the uncorrelated input and the uncorrelated output
        plt.figure()
        plt.grid(linewidth=0.4, color='k')
        plt.plot(c[:,0], c[:,1], 'ro', label='uncorrelated out')
        plt.plot(t, tt, 'bx', label='uncorrelated input')
        plt.title('Truncated-Gaussian: Comparison between input of direct and output of inverse')
        plt.axis('equal')
        plt.show()

        """
        def test_arcine(self):
        """#An arcisne distribution has the support over [a,b]
        """  
        def blackbox(x):
            return x 
        
        #   parameters:
        #   a real number
        #   b real number
        a   = 0.0
        b   = 1.0
        x = np.linspace(a, b, 100)
        # analytical mean and variance:
        a_mean     = (a+b)/2.0
        a_variance = (1.0/8.0)*(b-a)**2
        #
        print 'Test Chebychev:'
        print '--------------------------------------------'
        print 'analytical mean:', a_mean
        print 'analytical variance', a_variance 
                                                                                                                                                     
        # numerical solution using EffectiveQuadrature:
        xo = Parameter(order=5, distribution='Chebychev', lower = 0.010, upper =0.99)
        myBasis = Basis('Tensor')
        myPoly = Polyint([xo], myBasis)
        myPoly.computeCoefficients(blackbox)
        myStats = myPoly.getStatistics()
        print 'EffectiveQuadrature mean:', myStats.mean
        print 'EffectiveQuadrature variance:', myStats.variance
        
        # numerical solution using MonteCarlo:
        #N = 900000
        #xi = np.random.gamma(k,theta,(N,1))
        #yi = evalfunction(np.reshape(xi, (N,1)), blackbox)
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
        yo = Parameter(order = 5, distribution ='Chebychev', upper = 1.0, lower = 0.0 )
                                                                                                                                                     
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
        # distributions will be correlated
        # with the correlation defined in 
        # method 'CorrelationMatrix':
        obj = Nataf(D,R)
                                                                                                                                                     
        o  = obj.getCorrelatedSamples(N = 300)
        oo = obj.getUncorrelatedSamples(N = 300 )
        # correlated data:
        t   = o[:,0]
        tt  = o[:,1]
        
        # plot of the data:
        # correlated VS uncorrelated input
        plt.figure()
        plt.grid(linewidth= 0.4, color='k')
        plt.plot(t, tt, 'ro', label='new correlated')
        plt.plot(oo[:,0], oo[:,1], 'bo', label='original: uncorrelated')
        plt.legend(loc='upper left')
        plt.title('Chebychev: results of getCorrelatedSamples method')
        plt.axis('equal')
        plt.show()
        # check the mean and the variance after correlation:
        print '____ test the mean and the variance after getCorrelated____'
        print 'mean of uncorrelated inputs:', obj.D[0].mean, obj.D[1].mean
        print 'mean of correlated outputs:', np.mean(t), np.mean(tt)
        print 'variance of uncorrelated inputs:', obj.D[0].variance, obj.D[1].variance
        print 'variance of correlated outputs:', np.var(t), np.var(tt)
                                                                                                                                                     
        # testing direct transformation:
        u = obj.C2U(o)
        plt.figure()
        plt.grid(linewidth=0.4, color='k')
        plt.plot(u[:,0], u[:,1], 'ro', label='Uncorrelated outputs')
        plt.plot(t ,tt, 'bo', label='Correlated inputs')
        plt.legend(loc='upper left')
        plt.title('Nataf transformation for Chebychev distribution')
        plt.axis('equal')
        plt.show()
        
        # testing the inverse transformation:
        c = obj.U2C(u)
        plt.figure()
        plt.grid(linewidth=0.4, color='k')
        plt.plot(c[:,0], c[:,1], 'ro', label='Correlated output')
        plt.plot(u[:,0], u[:,1], 'bo', label='Uncorrelated input')
        plt.legend(loc='upper left')
        plt.title('Nataf inverse transformation for Chebychev distribution')
        plt.axis('equal')
        plt.show()
                                                                                                                                                     
        # check the uncorrelated input and the uncorrelated output
        plt.figure()
        plt.grid(linewidth=0.4, color='k')
        plt.plot(c[:,0], c[:,1], 'ro', label='uncorrelated out')
        plt.plot(t, tt, 'bx', label='uncorrelated input')
        plt.title('Chebychev: Comparison between input of direct and output of inverse')
        plt.axis('equal')
        plt.show()
        """
    
    def test_gaussian(self):                                                                                                                             
     """ A gaussian distribution has the support over (-inf, +inf)
     """  
     def blackbox(x):
         return x 
     
     #   parameters:
     #   variance > 0
     #   mean: real number
     var    = 2.5
     mean   = 14.0
     
     # analytical mean and variance:
     a_mean     = mean
     a_variance = var
     #
     print 'Test Gaussian:'
     print '--------------------------------------------'
     print 'analytical mean:', a_mean
     print 'analytical variance', a_variance 
                                                                                                                                                  
     # numerical solution using EffectiveQuadrature:
     xo = Parameter(order=5, distribution='gaussian', shape_parameter_A = mean, shape_parameter_B = var)
     myBasis = Basis('Tensor')
     myPoly = Polyint([xo], myBasis)
     myPoly.computeCoefficients(blackbox)
     myStats = myPoly.getStatistics()
     print 'EffectiveQuadrature mean:', myStats.mean
     print 'EffectiveQuadrature variance:', myStats.variance
     
     # numerical solution using MonteCarlo:
     N = 900000
     xi = np.random.normal(mean, np.sqrt(var),(N,1))
     yi = evalfunction(np.reshape(xi, (N,1)), blackbox)
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
     # distributions will be correlated
     # with the correlation defined in 
     # method 'CorrelationMatrix':
     obj = Nataf(D,R)
                                                                                                                                                  
     o  = obj.getCorrelatedSamples(N = 300)
     oo = obj.getUncorrelatedSamples(N = 300 )
     # correlated data:
     t   = o[:,0]
     tt  = o[:,1]
     
     # plot of the data:
     # correlated VS uncorrelated input
     plt.figure()
     plt.grid(linewidth= 0.4, color='k')
     plt.plot(t, tt, 'ro', label='new correlated')
     plt.plot(oo[:,0], oo[:,1], 'bo', label='original: uncorrelated')
     plt.legend(loc='upper left')
     plt.title('Gaussian: results of getCorrelatedSamples method')
     plt.axis('equal')
     plt.show()
     # check the mean and the variance after correlation:
     print '____ test the mean and the variance after getCorrelated____'
     print 'mean of uncorrelated inputs:', obj.D[0].mean, obj.D[1].mean
     print 'mean of correlated outputs:', np.mean(t), np.mean(tt)
     print 'variance of uncorrelated inputs:', obj.D[0].variance, obj.D[1].variance
     print 'variance of correlated outputs:', np.var(t), np.var(tt)
                                                                                                                                                  
     # testing direct transformation:
     u = obj.C2U(o)
     plt.figure()
     plt.grid(linewidth=0.4, color='k')
     plt.plot(u[:,0], u[:,1], 'ro', label='Uncorrelated outputs')
     plt.plot(t ,tt, 'bo', label='Correlated inputs')
     plt.legend(loc='upper left')
     plt.title('Nataf transformation for Gaussian distribution')
     plt.axis('equal')
     plt.show()
     
     # testing the inverse transformation:
     c = obj.U2C(u)
     plt.figure()
     plt.grid(linewidth=0.4, color='k')
     plt.plot(c[:,0], c[:,1], 'ro', label='Correlated output')
     plt.plot(u[:,0], u[:,1], 'bo', label='Uncorrelated input')
     plt.legend(loc='upper left')
     plt.title('Nataf inverse transformation for Gaussian distribution')
     plt.axis('equal')
     plt.show()
                                                                                                                                                  
     # check the uncorrelated input and the uncorrelated output
     plt.figure()
     plt.grid(linewidth=0.4, color='k')
     plt.plot(c[:,0], c[:,1], 'ro', label='uncorrelated out')
     plt.plot(t, tt, 'bx', label='uncorrelated input')
     plt.title('Gaussian: Comparison between input of direct and output of inverse')
     plt.axis('equal')
     plt.show()

    def test_rayleigh(self):
        """ A rayleigh distribution has the support over (-inf, +inf)
        """  
        def blackbox(x):
            return x 
        
        #   parameters:
        #   scale > 0
        #   
        scale    = 2.5
        
        # analytical mean and variance:
        a_mean     = scale*np.sqrt(np.pi/2.)
        a_variance = (4. - np.pi)/2. * scale**2
        #
        print 'Test Rayleigh:'
        print '--------------------------------------------'
        print 'analytical mean:', a_mean
        print 'analytical variance', a_variance 
                                                                                                                                                     
        # numerical solution using EffectiveQuadrature:
        xo = Parameter(order=5, distribution='rayleigh', shape_parameter_A = scale)
        myBasis = Basis('Tensor')
        myPoly = Polyint([xo], myBasis)
        myPoly.computeCoefficients(blackbox)
        myStats = myPoly.getStatistics()
        print 'EffectiveQuadrature mean:', myStats.mean
        print 'EffectiveQuadrature variance:', myStats.variance
        
        # numerical solution using MonteCarlo:
        N = 900000
        xi = np.random.rayleigh(scale,(N,1))
        yi = evalfunction(np.reshape(xi, (N,1)), blackbox)
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
        # distributions will be correlated
        # with the correlation defined in 
        # method 'CorrelationMatrix':
        obj = Nataf(D,R)
                                                                                                                                                     
        o  = obj.getCorrelatedSamples(N = 300)
        oo = obj.getUncorrelatedSamples(N = 300 )
        # correlated data:
        t   = o[:,0]
        tt  = o[:,1]
        
        # plot of the data:
        # correlated VS uncorrelated input
        plt.figure()
        plt.grid(linewidth= 0.4, color='k')
        plt.plot(t, tt, 'ro', label='new correlated')
        plt.plot(oo[:,0], oo[:,1], 'bo', label='original: uncorrelated')
        plt.legend(loc='upper left')
        plt.title('Rayleigh: results of getCorrelatedSamples method')
        plt.axis('equal')
        plt.show()
        # check the mean and the variance after correlation:
        print '____ test the mean and the variance after getCorrelated____'
        print 'mean of uncorrelated inputs:', obj.D[0].mean, obj.D[1].mean
        print 'mean of correlated outputs:', np.mean(t), np.mean(tt)
        print 'variance of uncorrelated inputs:', obj.D[0].variance, obj.D[1].variance
        print 'variance of correlated outputs:', np.var(t), np.var(tt)
                                                                                                                                                     
        # testing direct transformation:
        u = obj.C2U(o)
        plt.figure()
        plt.grid(linewidth=0.4, color='k')
        plt.plot(u[:,0], u[:,1], 'ro', label='Uncorrelated outputs')
        plt.plot(t ,tt, 'bo', label='Correlated inputs')
        plt.legend(loc='upper left')
        plt.title('Nataf transformation for Rayleigh distribution')
        plt.axis('equal')
        plt.show()
        
        # testing the inverse transformation:
        c = obj.U2C(u)
        plt.figure()
        plt.grid(linewidth=0.4, color='k')
        plt.plot(c[:,0], c[:,1], 'ro', label='Correlated output')
        plt.plot(u[:,0], u[:,1], 'bo', label='Uncorrelated input')
        plt.legend(loc='upper left')
        plt.title('Nataf inverse transformation for Rayleigh distribution')
        plt.axis('equal')
        plt.show()
                                                                                                                                                     
        # check the uncorrelated input and the uncorrelated output
        plt.figure()
        plt.grid(linewidth=0.4, color='k')
        plt.plot(c[:,0], c[:,1], 'ro', label='uncorrelated out')
        plt.plot(t, tt, 'bx', label='uncorrelated input')
        plt.title('Rayleigh: Comparison between input of direct and output of inverse')
        plt.axis('equal')
        plt.show()

    def test_uniform(self):
        """ A uniform distribution has the support over (-inf, +inf)
        """  
        def blackbox(x):
            return x 
        
        #   parameters:
        #   scale > 0
        #   
        lower = -1.0
        upper = 1.0
        # analytical mean and variance:
        a_mean     = 0.5*(lower + upper)
        a_variance = 1. / 12. *(upper -lower)**2
        #
        print 'Test Uniform:'
        print '--------------------------------------------'
        print 'analytical mean:', a_mean
        print 'analytical variance', a_variance 
                                                                                                                                                     
        # numerical solution using EffectiveQuadrature:
        xo = Parameter(order=5, distribution='uniform', lower = lower, upper = upper )
        myBasis = Basis('Tensor')
        myPoly = Polyint([xo], myBasis)
        myPoly.computeCoefficients(blackbox)
        myStats = myPoly.getStatistics()
        print 'EffectiveQuadrature mean:', myStats.mean
        print 'EffectiveQuadrature variance:', myStats.variance
        
        # numerical solution using MonteCarlo:
        N = 900000
        xi = np.random.uniform(lower, upper ,(N,1))
        yi = evalfunction(np.reshape(xi, (N,1)), blackbox)
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
        # distributions will be correlated
        # with the correlation defined in 
        # method 'CorrelationMatrix':
        obj = Nataf(D,R)
                                                                                                                                                     
        o  = obj.getCorrelatedSamples(N = 300)
        oo = obj.getUncorrelatedSamples(N = 300 )
        # correlated data:
        t   = o[:,0]
        tt  = o[:,1]
        
        # plot of the data:
        # correlated VS uncorrelated input
        plt.figure()
        plt.grid(linewidth= 0.4, color='k')
        plt.plot(t, tt, 'ro', label='new correlated')
        plt.plot(oo[:,0], oo[:,1], 'bo', label='original: uncorrelated')
        plt.legend(loc='upper left')
        plt.title('Uniform: results of getCorrelatedSamples method')
        plt.axis('equal')
        plt.show()
        # check the mean and the variance after correlation:
        print '____ test the mean and the variance after getCorrelated____'
        print 'mean of uncorrelated inputs:', obj.D[0].mean, obj.D[1].mean
        print 'mean of correlated outputs:', np.mean(t), np.mean(tt)
        print 'variance of uncorrelated inputs:', obj.D[0].variance, obj.D[1].variance
        print 'variance of correlated outputs:', np.var(t), np.var(tt)
                                                                                                                                                     
        # testing direct transformation:
        u = obj.C2U(o)
        plt.figure()
        plt.grid(linewidth=0.4, color='k')
        plt.plot(u[:,0], u[:,1], 'ro', label='Uncorrelated outputs')
        plt.plot(t ,tt, 'bo', label='Correlated inputs')
        plt.legend(loc='upper left')
        plt.title('Nataf transformation for Uniform distribution')
        plt.axis('equal')
        plt.show()
        
        # testing the inverse transformation:
        c = obj.U2C(u)
        plt.figure()
        plt.grid(linewidth=0.4, color='k')
        plt.plot(c[:,0], c[:,1], 'ro', label='Correlated output')
        plt.plot(u[:,0], u[:,1], 'bo', label='Uncorrelated input')
        plt.legend(loc='upper left')
        plt.title('Nataf inverse transformation for Uniform distribution')
        plt.axis('equal')
        plt.show()
                                                                                                                                                     
        # check the uncorrelated input and the uncorrelated output
        plt.figure()
        plt.grid(linewidth=0.4, color='k')
        plt.plot(c[:,0], c[:,1], 'ro', label='uncorrelated out')
        plt.plot(t, tt, 'bx', label='uncorrelated input')
        plt.title('Uniform: Comparison between input of direct and output of inverse')
        plt.axis('equal')
        plt.show()

    def test_chisquared(self):                                                                                                                              
       """ A chisquared distribution has the support over (0, +inf)
           and [0, +inf) when k ==1.
       """  
       def blackbox(x):
           return x 
       
       #   parameters:
       #   k : degree of freedom
       #   
       k = 15
       
       # analytical mean and variance:
       a_mean     = k
       a_variance = 2. * k
       #
       print 'Test Uniform:'
       print '--------------------------------------------'
       print 'analytical mean:', a_mean
       print 'analytical variance', a_variance 
                                                                                                                                                    
       # numerical solution using EffectiveQuadrature:
       xo = Parameter(order=5, distribution='Chisquared', shape_parameter_A = k )
       myBasis = Basis('Tensor')
       myPoly = Polyint([xo], myBasis)
       myPoly.computeCoefficients(blackbox)
       myStats = myPoly.getStatistics()
       print 'EffectiveQuadrature mean:', myStats.mean
       print 'EffectiveQuadrature variance:', myStats.variance
       
       # numerical solution using MonteCarlo:
       N = 900000
       xi = np.random.chisquare(k ,(N,1))
       yi = evalfunction(np.reshape(xi, (N,1)), blackbox)
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
       # distributions will be correlated
       # with the correlation defined in 
       # method 'CorrelationMatrix':
       obj = Nataf(D,R)
                                                                                                                                                    
       o  = obj.getCorrelatedSamples(N = 300)
       oo = obj.getUncorrelatedSamples(N = 300 )
       # correlated data:
       t   = o[:,0]
       tt  = o[:,1]
       
       # plot of the data:
       # correlated VS uncorrelated input
       plt.figure()
       plt.grid(linewidth= 0.4, color='k')
       plt.plot(t, tt, 'ro', label='new correlated')
       plt.plot(oo[:,0], oo[:,1], 'bo', label='original: uncorrelated')
       plt.legend(loc='upper left')
       plt.title('Chisquared: results of getCorrelatedSamples method')
       plt.axis('equal')
       plt.show()
       # check the mean and the variance after correlation:
       print '____ test the mean and the variance after getCorrelated____'
       print 'mean of uncorrelated inputs:', obj.D[0].mean, obj.D[1].mean
       print 'mean of correlated outputs:', np.mean(t), np.mean(tt)
       print 'variance of uncorrelated inputs:', obj.D[0].variance, obj.D[1].variance
       print 'variance of correlated outputs:', np.var(t), np.var(tt)
                                                                                                                                                    
       # testing direct transformation:
       u = obj.C2U(o)
       plt.figure()
       plt.grid(linewidth=0.4, color='k')
       plt.plot(u[:,0], u[:,1], 'ro', label='Uncorrelated outputs')
       plt.plot(t ,tt, 'bo', label='Correlated inputs')
       plt.legend(loc='upper left')
       plt.title('Nataf transformation for Chisquared distribution')
       plt.axis('equal')
       plt.show()
       
       # testing the inverse transformation:
       c = obj.U2C(u)
       plt.figure()
       plt.grid(linewidth=0.4, color='k')
       plt.plot(c[:,0], c[:,1], 'ro', label='Correlated output')
       plt.plot(u[:,0], u[:,1], 'bo', label='Uncorrelated input')
       plt.legend(loc='upper left')
       plt.title('Nataf inverse transformation for Chisquared distribution')
       plt.axis('equal')
       plt.show()
                                                                                                                                                    
       # check the uncorrelated input and the uncorrelated output
       plt.figure()
       plt.grid(linewidth=0.4, color='k')
       plt.plot(c[:,0], c[:,1], 'ro', label='uncorrelated out')
       plt.plot(t, tt, 'bx', label='uncorrelated input')
       plt.title('Chisquared: Comparison between input of direct and output of inverse')
       plt.axis('equal')
       plt.show()


    def test_exponential(self):                                                                                                                              
       """ An exponential distribution has the support over [0, +inf)
       """  
       def blackbox(x):
           return x 
       
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
       #
       print 'Test Exponential:'
       print '--------------------------------------------'
       print 'analytical mean:', a_mean
       print 'analytical variance', a_variance 
                                                                                                                                                    
       # numerical solution using EffectiveQuadrature:
       xo = Parameter(order=5, distribution='Exponential', shape_parameter_A = 1./scale )
       myBasis = Basis('Tensor')
       myPoly = Polyint([xo], myBasis) 
       myPoly.computeCoefficients(blackbox)
       myStats = myPoly.getStatistics()
       print 'EffectiveQuadrature mean:', myStats.mean
       print 'EffectiveQuadrature variance:', myStats.variance
       
       # numerical solution using MonteCarlo:
       N = 900000
       xi = np.random.exponential(1./scale ,(N,1))
       yi = evalfunction(np.reshape(xi, (N,1)), blackbox)
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
       # distributions will be correlated
       # with the correlation defined in 
       # method 'CorrelationMatrix':
       obj = Nataf(D,R)
                                                                                                                                                    
       o  = obj.getCorrelatedSamples(N = 300)
       oo = obj.getUncorrelatedSamples(N = 300 )
       # correlated data:
       t   = o[:,0]
       tt  = o[:,1]
       
       # plot of the data:
       # correlated VS uncorrelated input
       plt.figure()
       plt.grid(linewidth= 0.4, color='k')
       plt.plot(t, tt, 'ro', label='new correlated')
       plt.plot(oo[:,0], oo[:,1], 'bo', label='original: uncorrelated')
       plt.legend(loc='upper left')
       plt.title('Exponential: results of getCorrelatedSamples method')
       plt.axis('equal')
       plt.show()
       # check the mean and the variance after correlation:
       print '____ test the mean and the variance after getCorrelated____'
       print 'mean of uncorrelated inputs:', obj.D[0].mean, obj.D[1].mean
       print 'mean of correlated outputs:', np.mean(t), np.mean(tt)
       print 'variance of uncorrelated inputs:', obj.D[0].variance, obj.D[1].variance
       print 'variance of correlated outputs:', np.var(t), np.var(tt)
                                                                                                                                                    
       # testing direct transformation:
       u = obj.C2U(o)
       plt.figure()
       plt.grid(linewidth=0.4, color='k')
       plt.plot(u[:,0], u[:,1], 'ro', label='Uncorrelated outputs')
       plt.plot(t ,tt, 'bo', label='Correlated inputs')
       plt.legend(loc='upper left')
       plt.title('Exponential transformation for Chisquared distribution')
       plt.axis('equal')
       plt.show()
       
       # testing the inverse transformation:
       c = obj.U2C(u)
       plt.figure()
       plt.grid(linewidth=0.4, color='k')
       plt.plot(c[:,0], c[:,1], 'ro', label='Correlated output')
       plt.plot(u[:,0], u[:,1], 'bo', label='Uncorrelated input')
       plt.legend(loc='upper left')
       plt.title('Exponential inverse transformation for Chisquared distribution')
       plt.axis('equal')
       plt.show()
                                                                                                                                                    
       # check the uncorrelated input and the uncorrelated output
       plt.figure()
       plt.grid(linewidth=0.4, color='k')
       plt.plot(c[:,0], c[:,1], 'ro', label='uncorrelated out')
       plt.plot(t, tt, 'bx', label='uncorrelated input')
       plt.title('Exponential: Comparison between input of direct and output of inverse')
       plt.axis('equal')
       plt.show()
def testbasic(self):
    print 'done!'
                                           
if __name__== '__main__':
    unittest.main()
