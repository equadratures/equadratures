from unittest import TestCase
import unittest
from equadratures import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf, gamma
N = 900000

def plot(x, f_X, a, b, samples, dist_name, ylims=None):
  fig = plt.figure()
  ax = fig.add_subplot(1,1,1)
  plt.hist(samples, 30, normed=1, facecolor='orangered', edgecolor='black', linewidth=0.5, label='parameter-samples')
  plt.plot(x, f_X, c='navy', lw=3, label='input')
  plt.plot(a, b, '--', c='green', label='parameter-pdf')
  adjust_spines(ax, ['left', 'bottom'])
  legend = ax.legend(loc='upper left')
  plt.xlabel('Viscosity', fontsize=13)
  plt.ylabel('PDF', fontsize=13)
  plt.xticks(fontsize=13)
  plt.yticks(fontsize=13)
  plt.title(dist_name)
  if ylims is not None:
    plt.ylim([ylims[0], ylims[1]])
  ax.set_xlabel('X variable')
  ax.set_ylabel('PDF')
  plt.tight_layout()
  plt.show()
def blackbox(x):
  return x
class Test_Distributions(TestCase): 
    def test_gamma(self):
       x = np.linspace(0, 20, 100)
       k = 2.0
       theta = 0.9
       # the following costant is the analytica solution of integral_(0)^(\infty) ( x^1 * e^(-x) )
       c_1 = 1.0
       mean_5 = k*theta
       variance_5 = k*theta**2
       f_X = np.zeros(len(x))
       for i in range(0,len(x)):
           f_X[i] = (1/c_1)*(1/theta**k)*((x[i])**(k-1))*np.exp(-x[i]/theta)
       xo = Parameter(order=1, distribution='Gamma',shape_parameter_A = 2.0, shape_parameter_B = 0.9 )
       myBasis = Basis('univariate')
       myPoly = Polyint([xo], myBasis)
       myPoly.computeCoefficients(blackbox)
       myStats = myPoly.getStatistics()
       a,b = xo.getPDF(points=150)
       samples = xo.getSamples(m=1000)
       std_dev = np.std(samples)
       plot(x, f_X, a, b, samples, xo.name)
       xi = np.random.gamma(k,theta, (N,1))
       yi = evalfunction(np.reshape(xi, (N, 1) ), blackbox)
       print xo.name, xo.shape_parameter_A, xo.shape_parameter_B
       print 'Effective quadrature mean: ', myStats.mean#, myStats.variance
       print 'Effective quadrature variance:' , myStats.variance
       print 'Monte Carlo mean:', np.mean(yi)
       print 'Monte Carlo variance:', np.var(yi)
       print 'Analytical mean :', xo.mean 
       print 'Analytical variance :', xo.variance
       print '\n'
       eq_m = float('%.4f' %myStats.mean)
       mc_m = float('%.4f' %np.mean(yi))
       error_mean = np.testing.assert_almost_equal(eq_m, mc_m, decimal=2, err_msg="difference greater than imposed tolerance for mean value")
       eq_v = float('%.4f' %myStats.variance)
       mc_v = float('%.4f' %np.var(yi))
       error_var = np.testing.assert_almost_equal(eq_v, mc_v, decimal=1, err_msg="difference greater than imposed tolerance for variance value")
    def test_beta(self):
       shape_A = 2.0 # alpha
       shape_B = 3.0 # beta
       x = np.linspace(0,1,100)
       mean_3  = shape_A/(shape_A + shape_B)
       variance_3 = (shape_A* shape_B)/((shape_A+shape_B+1)*(shape_A+shape_B)**2)
       c_1= 1.0/12
       c_2 = 2.5058
       f_X = np.zeros(len(x))
       for i in range(0,len(x)):
           f_X[i] = (1.0/c_1)* ((x[i])**(shape_A-1))*(1-x[i])**(shape_B-1)
       xo = Parameter(order=1, distribution='Beta',lower =0.0, upper=1.0, shape_parameter_A = shape_A, shape_parameter_B = shape_B )
       myBasis = Basis('univariate')
       myPoly = Polyint([xo], myBasis)
       myPoly.computeCoefficients(blackbox)
       myStats = myPoly.getStatistics()                                                                     
       xi = np.random.beta(shape_A, shape_B, (N,1))
       yi = evalfunction(np.reshape(xi, (N, 1) ), blackbox)
       a,b = xo.getPDF(points=150)
       samples = xo.getSamples(m=1000)
       plot(x, f_X, a, b, samples, xo.name)
       print xo.name, xo.shape_parameter_A, xo.shape_parameter_B
       print 'Effective quadrature mean: ', myStats.mean#, myStats.variance
       print 'Effective quadrature variance:' , myStats.variance
       print 'Monte Carlo mean:', np.mean(yi)
       print 'Monte Carlo variance:', np.var(yi)
       print 'Analytical mean :', xo.mean 
       print 'Analytical variance :', xo.variance
       print '\n'
       eq_m = float('%.4f' %myStats.mean)
       mc_m = float('%.4f' %np.mean(yi))
       error_mean = np.testing.assert_almost_equal(eq_m, mc_m, decimal=2, err_msg="difference greater than imposed tolerance for mean value")
       eq_v = float('%.4f' %myStats.variance)
       mc_v = float('%.4f' %np.var(yi))
       error_var = np.testing.assert_almost_equal(eq_v, mc_v, decimal=1, err_msg="difference greater than imposed tolerance for variance value")
    def test_weibull(self):
       x = np.linspace(10**(-10),30,100)
       lambdaa = 2.0 # SCALE OF DISTRIBUTION
       k = 5.0 # SHAPE OF DISTRIBUTION
       f_X = np.zeros(len(x))
       for i in range(0,len(x)):
           f_X[i] = (k/lambdaa)*((x[i]/lambdaa)**(k-1))* np.exp(-(x[i]/lambdaa)**k)
       xo = Parameter(order=1, distribution='Weibull', shape_parameter_A =lambdaa , shape_parameter_B = k)
       myBasis = Basis('univariate')
       myPoly = Polyint([xo], myBasis)
       myPoly.computeCoefficients(blackbox)
       myStats = myPoly.getStatistics()                                                                                                                   
       xi_o = np.random.rand(N,1)
       xi = lambdaa * (-np.log(xi_o))**(1.0/k)
       yi = evalfunction(np.reshape(xi, (N, 1) ), blackbox)
       a,b = xo.getPDF(points=150)
       samples = xo.getSamples(m=1000)
       plot(x, f_X, a, b, samples, xo.name)
       print xo.name, xo.shape_parameter_A, xo.shape_parameter_B
       print 'Effective quadrature mean: ', myStats.mean#, myStats.variance
       print 'Effective quadrature variance:' , myStats.variance
       print 'Monte Carlo mean:', np.mean(yi)
       print 'Monte Carlo variance:', np.var(yi)
       print 'Analytical mean :', xo.mean 
       print 'Analytical variance :', xo.variance
       print '\n'
       eq_m = float('%.4f' %myStats.mean)
       mc_m = float('%.4f' %np.mean(yi))
       error_mean = np.testing.assert_almost_equal(eq_m, mc_m, decimal=2, err_msg="difference greater than imposed tolerance for mean value")
       eq_v = float('%.4f' %myStats.variance)
       mc_v = float('%.4f' %np.var(yi))
       error_var = np.testing.assert_almost_equal(eq_v, mc_v, decimal=1, err_msg="difference greater than imposed tolerance for variance value")
    def test_truncated_gauss(self):
      mu = 1.0  # mean'
      sigma = 0.5 # std deviation'
      a = -2.0
      b = 2.0
      alpha = (a-mu)/ sigma
      beta = (b-mu)/sigma
      Z = 0.5*(1.0+erf(beta/np.sqrt(2.0)))-0.5*(1.0+erf(alpha/np.sqrt(2.0)))
      x = np.linspace(a,b,150)
      f_X = np.linspace(0,1,150)
      for i in range(0,len(x)):
          xi = (x[i]-mu)/sigma
          phi_zeta = (1.0/(np.sqrt(2.0*np.pi)))*np.exp(-0.5*xi**2)
          f_X[i] = phi_zeta/(sigma*Z) 
      phi_alpha = (1.0/(np.sqrt(2.0*np.pi)))*np.exp(-0.5*alpha**2)
      phi_beta = (1.0/(np.sqrt(2.0*np.pi)))*np.exp(-0.5*beta**2)
      mean = mu + (phi_alpha - phi_beta)*sigma/Z
      variance = (sigma**2)* (1+  (alpha*phi_alpha - beta*phi_beta)/Z  -((alpha*phi_alpha - beta*phi_beta)/Z )**2)
        
      shape_A = mu
      shape_B = sigma**2
      xo = Parameter(order=1, distribution='truncated-gaussian',lower =a, upper=b, shape_parameter_A = shape_A, shape_parameter_B = shape_B )
      myBasis = Basis('univariate')
      myPoly = Polyint([xo], myBasis)
      myPoly.computeCoefficients(blackbox)
      myStats = myPoly.getStatistics()
      a,b = xo.getPDF(points=150) # analytical!
      samples = xo.getSamples(m=3000)
      yi = samples
      plot(x, f_X, a, b, samples, xo.name)
      print xo.name, xo.shape_parameter_A, xo.shape_parameter_B, xo.lower, xo.upper
      print 'Effective quadrature mean: ', myStats.mean#, myStats.variance
      print 'Effective quadrature variance:' , myStats.variance
      print 'Analytical mean :', xo.mean 
      print 'Analytical variance :', xo.variance
      print 'Monte Carlo mean:', np.mean(yi)
      print 'Monte Carlo variance:', np.var(yi)
      print '\n'
      eq_m = float('%.4f' %myStats.mean)
      mc_m = float('%.4f' %np.mean(yi))
      error_mean = np.testing.assert_almost_equal(eq_m, xo.mean, decimal=1, err_msg="difference greater than imposed tolerance for mean value")
      eq_v = float('%.4f' %myStats.variance)
      mc_v = float('%.4f' %np.var(yi))
      error_var = np.testing.assert_almost_equal(eq_v, xo.variance, decimal=1, err_msg="difference greater than imposed tolerance for variance value")
    def test_arcsine(self):
      a = 0.0#0.001
      b = 1.0#0.99
      x = np.linspace(a, b, 100) # domain for Chebyshev
      mean_1 = (a+b)/2.0 
      variance_1 = (1.0/8.0)*(b-a)**2
      f_X= np.zeros(len(x))

      for i in range(0,len(x)):
        if x[i] == a :
           f_X[i] =  1.0/(np.pi* np.sqrt(((x[i]+0.000000001 )- a)*(b - (x[i]-0.000000001)) ))
        elif x[i] == b:
           f_X[i] =  1.0/(np.pi* np.sqrt(((x[i]+0.000000001 )- a)*(b - (x[i]-0.000000001)) ))
        else: 
           f_X[i] = 1.0/(np.pi* np.sqrt((x[i] - a)*(b - x[i])) )
      xo = Parameter(order=1, distribution='Chebyshev',lower =0.001, upper=0.99)
      myBasis = Basis('univariate')
      myPoly = Polyint([xo], myBasis)
      myPoly.computeCoefficients(blackbox)
      myStats = myPoly.getStatistics()
      a,b = xo.getPDF(N=150) # analytical!
      samples = xo.getSamples(m=3000)
      yi = samples
      plot(x, f_X, a, b, samples, xo.name, ylims=[0., 3.])
      print xo.name, xo.shape_parameter_A, xo.shape_parameter_B, xo.lower, xo.upper
      print 'Effective quadrature mean: ', myStats.mean#, myStats.variance
      print 'Effective quadrature variance:' , myStats.variance
      print 'Analytical mean :', xo.mean 
      print 'Analytical variance :', xo.variance
      print 'Monte Carlo mean:', np.mean(yi)
      print 'Monte Carlo variance:', np.var(yi)
      print '\n'
      eq_m = float('%.4f' %myStats.mean)
      mc_m = float('%.4f' %np.mean(yi))
      error_mean = np.testing.assert_almost_equal(eq_m, xo.mean, decimal=1, err_msg="difference greater than imposed tolerance for mean value")
      eq_v = float('%.4f' %myStats.variance)
      mc_v = float('%.4f' %np.var(yi))
      error_var = np.testing.assert_almost_equal(eq_v, xo.variance, decimal=1, err_msg="difference greater than imposed tolerance for variance value")
    def test_cauchy(self):
      x0 = 0.0
      gamma = 0.5
      x = np.linspace(-15.,15.,100)
      f_X = np.zeros(len(x))
      for i in range(0,len(x)):
        f_X[i] = 1.0/(np.pi*gamma*(1+(((x[i]-x0)/gamma)**2)))
      xo = Parameter(order=1, distribution='Cauchy', shape_parameter_A = x0, shape_parameter_B = gamma )
      myBasis = Basis('univariate')
      myPoly = Polyint([xo], myBasis)
      myPoly.computeCoefficients(blackbox)
      myStats = myPoly.getStatistics()
      a,b = xo.getPDF(points=150)
      samples = xo.getSamples(m=1000)
      yi = samples
      plot(x, f_X, a, b, samples, xo.name, ylims=[0., 0.8])
      print xo.name, xo.shape_parameter_A, xo.shape_parameter_B
      print 'Effective quadrature mean: ', myStats.mean#, myStats.variance
      print 'Effective quadrature variance:' , myStats.variance
      print 'Monte Carlo mean:', np.mean(yi)
      print 'Monte Carlo variance:', np.var(yi)
      print 'Analytical mean :', xo.mean 
      print 'Analytical variance :', xo.variance
      print '\n'

if __name__ == '__main__':
    unittest.main()
