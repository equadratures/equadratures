from unittest import TestCase
import unittest
from equadratures import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf, gamma
N = 900000
def blackbox(x):
  return x
class Test_Distributions(TestCase):
    def test_gamma(self):
       x = np.linspace(0, 20, 100)
       k = 2.0
       theta = 0.9
       # the following costant is the analytical solution of integral_(0)^(\infty) ( x^1 * e^(-x) )
       c_1 = 1.0
       mean_5 = k*theta
       variance_5 = k*theta**2
       f_X = np.zeros(len(x))

       for i in range(0,len(x)):
           f_X[i] = (1/c_1)*(1/theta**k)*((x[i])**(k-1))*np.exp(-x[i]/theta)

       xo = Parameter(order=1, distribution='Gamma',shape_parameter_A = 2.0, shape_parameter_B = 0.9 )
       myBasis = Basis('univariate')
       myPoly = Poly(xo, myBasis, method='numerical-integration')
       myPoly.set_model(blackbox)
       mean, variance = myPoly.get_mean_and_variance()
       samples = xo.get_samples(1000)
       std_dev = np.std(samples)
       xi = np.random.gamma(k,theta, (N,1))
       yi = evaluate_model(np.reshape(xi, (N, 1) ), blackbox)
       eq_m = float('%.4f' %mean)
       mc_m = float('%.4f' %np.mean(yi))
       error_mean = np.testing.assert_almost_equal(eq_m, mc_m, decimal=2, err_msg="difference greater than imposed tolerance for mean value")
       eq_v = float('%.4f' %variance)
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
       myPoly = Poly(xo, myBasis, method='numerical-integration')
       myPoly.set_model(blackbox)
       mean, variance = myPoly.get_mean_and_variance()
       xi = np.random.beta(shape_A, shape_B, (N,1))
       yi = evaluate_model(np.reshape(xi, (N, 1) ), blackbox)
       samples = xo.get_samples(1000)
       eq_m = float('%.4f' %mean)
       mc_m = float('%.4f' %np.mean(yi))
       error_mean = np.testing.assert_almost_equal(eq_m, mc_m, decimal=2, err_msg="difference greater than imposed tolerance for mean value")
       eq_v = float('%.4f' %variance)
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
       myPoly = Poly(xo, myBasis, method='numerical-integration')
       myPoly.set_model(blackbox)
       mean, variance = myPoly.get_mean_and_variance()
       xi_o = np.random.rand(N,1)
       xi = lambdaa * (-np.log(xi_o))**(1.0/k)
       yi = evaluate_model(np.reshape(xi, (N, 1) ), blackbox)
       samples = xo.get_samples(1000)
       eq_m = float('%.4f' %mean)
       mc_m = float('%.4f' %np.mean(yi))
       error_mean = np.testing.assert_almost_equal(eq_m, mc_m, decimal=2, err_msg="difference greater than imposed tolerance for mean value")
       eq_v = float('%.4f' %variance)
       mc_v = float('%.4f' %np.var(yi))
       error_var = np.testing.assert_almost_equal(eq_v, mc_v, decimal=1, err_msg="difference greater than imposed tolerance for variance value")
    def test_truncated_gauss(self):
      mu = 1.0  # mean'
      sigma = 0.5 # std deviation'
      a = -2.0
      b = 2.0
      alpha = (a-mu)/ sigma
      beta = (b-mu)/sigma
      den = 1. / (sigma*np.sqrt(2.* np.pi))
      x = np.linspace(a,b,150)
      f_X = np.linspace(0,1,150)
      parent_pdf = np.linspace(0,1,150)
      cdf_parent = np.linspace(0,1,150)
      for i in range(0,len(x)):
          num = np.exp(-(x[i]-mu)**2 / (2. * sigma**2))
          xi = (x[i]-mu)/sigma
          #xi = x
          parent_pdf[i] = num * den
          u = (b - mu)/sigma
          l = (a - mu)/sigma
          cdf_parentb = 0.5*(1+ erf(b / np.sqrt(2)))
          cdf_parenta = 0.5*(1+ erf(a / np.sqrt(2)))
          cdf_parent = cdf_parentb - cdf_parenta
          f_X[i] = parent_pdf[i]/cdf_parent

      mean = mu
      variance = sigma**2
      shape_A = mu
      shape_B = sigma**2
      xo = Parameter(order=1, distribution='truncated-gaussian',lower =a, upper=b, shape_parameter_A = shape_A, shape_parameter_B = shape_B )
      myBasis = Basis('univariate')
      myPoly = Poly(xo, myBasis, method='numerical-integration')
      myPoly.set_model(blackbox)
      mean, variance = myPoly.get_mean_and_variance()
      a = x
      b = xo.get_pdf(a) # analytical!
      samples = xo.get_samples(3000)
      yi = samples
      eq_m = float('%.4f' %mean)
      mc_m = float('%.4f' %np.mean(yi))
      error_mean = np.testing.assert_almost_equal(eq_m, xo.mean, decimal=1, err_msg="difference greater than imposed tolerance for mean value")
      eq_v = float('%.4f' %variance)
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
      myPoly = Poly(xo, myBasis, method='numerical-integration')
      myPoly.set_model(blackbox)
      mean, variance = myPoly.get_mean_and_variance()
      a = x
      b = xo.get_pdf(a) # analytical!
      samples = xo.get_samples(3000)
      yi = samples
      eq_m = float('%.4f' %mean)
      mc_m = float('%.4f' %np.mean(yi))
      error_mean = np.testing.assert_almost_equal(eq_m, xo.mean, decimal=1, err_msg="difference greater than imposed tolerance for mean value")
      eq_v = float('%.4f' %variance)
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
      myPoly = Poly(xo, myBasis, method='numerical-integration')
      myPoly.set_model(blackbox)
      mean, variance = myPoly.get_mean_and_variance()
      a = x
      b = xo.get_pdf(a)
      samples = xo.get_samples(1000)
      yi = samples
    def test_exponential(self):
      param = Parameter(order=1, distribution='exponential', shape_parameter_A=2.0)
      print(param.mean)
      s_values, pdf = param.get_pdf()
      s_values, cdf = param.get_cdf()
      s_samples = param.get_samples(6000)
      param.get_description()
      s_samples = param.get_icdf(np.linspace(0., 1., 30))
      myBasis = Basis('univariate')
      myPoly = Poly(param, myBasis, method='numerical-integration')
      myPoly.set_model(blackbox)
      mean, variance = myPoly.get_mean_and_variance()
      np.testing.assert_almost_equal(mean, 1/param.shape_parameter_A, decimal=1)
      np.testing.assert_almost_equal(variance, 1/(param.shape_parameter_A)**2, decimal=1)
    def test_chisquared(self):
      param = Parameter(order=1, distribution='chi-squared', shape_parameter_A=2.0)
      s_values, pdf = param.get_pdf()
      s_values, cdf = param.get_cdf()
      s_samples = param.get_samples(6000)
      param.get_description()
      s_samples = param.get_icdf(np.linspace(0., 1., 30))
      myBasis = Basis('univariate')
      myPoly = Poly(param, myBasis, method='numerical-integration')
      myPoly.set_model(blackbox)
      mean, variance = myPoly.get_mean_and_variance()
      np.testing.assert_almost_equal(mean, param.shape_parameter_A, decimal=1)
      np.testing.assert_almost_equal(variance, 2.0 * param.shape_parameter_A, decimal=1)
    def test_chi(self):
      param = Parameter(order=1, distribution='chi', shape_parameter_A=2.0)
      s_values, pdf = param.get_pdf()
      s_values, cdf = param.get_cdf()
      s_samples = param.get_samples(6000)
      param.get_description()
      s_samples = param.get_icdf(np.linspace(0., 1., 30))
      myBasis = Basis('univariate')
      myPoly = Poly(param, myBasis, method='numerical-integration')
      myPoly.set_model(blackbox)
      mean, variance = myPoly.get_mean_and_variance()
      np.testing.assert_almost_equal(mean, param.mean, decimal=2)
      np.testing.assert_almost_equal(variance, param.variance, decimal=2)
    def test_pareto(self):
      param = Parameter(order=1, distribution='pareto', shape_parameter_A=2.0)
      s_values, pdf = param.get_pdf()
      s_values, cdf = param.get_cdf()
      s_samples = param.get_samples(6000)
      param.get_description()
      s_samples = param.get_icdf(np.linspace(0., 1., 30))
      myBasis = Basis('univariate')
      myPoly = Poly(param, myBasis, method='numerical-integration')
      myPoly.set_model(blackbox)
      mean, variance = myPoly.get_mean_and_variance()
      np.testing.assert_almost_equal(mean, param.mean, decimal=1)
    def test_gumbel(self):
      param = Parameter(order=1, distribution='gumbel', shape_parameter_A=2.0, shape_parameter_B=3.2)
      s_values, pdf = param.get_pdf()
      s_values, cdf = param.get_cdf()
      s_samples = param.get_samples(6000)
      param.get_description()
      s_samples = param.get_icdf(np.linspace(0., 1., 30))
      myBasis = Basis('univariate')
      myPoly = Poly(param, myBasis, method='numerical-integration')
      myPoly.set_model(blackbox)
      mean, variance = myPoly.get_mean_and_variance()
      np.testing.assert_almost_equal(mean, param.mean, decimal=1)
    def test_logistic(self):
      return 0
    def test_studentt(self):
      return 0
    def test_lognormal(self):
      return 0
    def test_rayleigh(self):
      param = Parameter(order=1, distribution='rayleigh', shape_parameter_A=2.0)
      s_values, pdf = param.get_pdf()
      s_values, cdf = param.get_cdf()
      s_samples = param.get_samples(6000)
      param.get_description()
      s_samples = param.get_icdf(np.linspace(0., 1., 30))
      myBasis = Basis('univariate')
      myPoly = Poly(param, myBasis, method='numerical-integration')
      myPoly.set_model(blackbox)
      mean, variance = myPoly.get_mean_and_variance()
      np.testing.assert_almost_equal(mean, param.mean, decimal=2)
      np.testing.assert_almost_equal(variance, param.variance, decimal=2)
      myPoly.get_summary()
    def test_custom(self):
      paramtest = Parameter(order=1, distribution='gaussian', shape_parameter_A=3, shape_parameter_B=0.5)
      stest_samples = paramtest.get_samples(6000)
      param = Parameter(order=1, distribution='custom', data=stest_samples)
      s_values, pdf = param.get_pdf()
      s_values, cdf = param.get_cdf()
      s_samples = param.get_samples(6000)
      param.get_description()
      s_samples = param.get_icdf(np.linspace(0., 1., 30))
      myBasis = Basis('univariate')
      myPoly = Poly(param, myBasis, method='numerical-integration')
      myPoly.set_model(blackbox)
      mean, variance = myPoly.get_mean_and_variance()
      np.testing.assert_almost_equal(mean, paramtest.shape_parameter_A, decimal=1)
      np.testing.assert_almost_equal(variance, paramtest.shape_parameter_B, decimal=1)
if __name__ == '__main__':
    unittest.main()
