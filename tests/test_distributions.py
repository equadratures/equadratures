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

       params = [
        Parameter(order=1, distribution='Gamma', shape_parameter_A = 2.0, shape_parameter_B = 0.9 ),
        Parameter(order=1, distribution='Gamma', shape = 2.0, scale = 0.9 ),
        Parameter(order=1, distribution='Gamma', k = 2.0, theta = 0.9 ),
        Gamma(order=1, shape_parameter_A = 2.0, shape_parameter_B = 0.9 ),
        Gamma(order=1, shape = 2.0, scale = 0.9 ),
        Gamma(order=1, k = 2.0, theta = 0.9 ),
       ]
       for param in params:
        myBasis = Basis('univariate')
        myPoly = Poly(param, myBasis, method='numerical-integration')
        myPoly.set_model(blackbox)
        mean, variance = myPoly.get_mean_and_variance()
        param.get_description()
        s_values, pdf = param.get_pdf()
        s_values, cdf = param.get_cdf()
        s_samples = param.get_samples(6000)
        param.get_description()
        s_samples = param.get_icdf(np.linspace(0., 1., 30))
        samples = param.get_samples(1000)
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
       xo = Parameter(order=2, distribution='Beta', lower=0.0, upper=1.0, shape_parameter_A=shape_A, shape_parameter_B=shape_B )
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
       lamda = 2.0 # SCALE OF DISTRIBUTION
       k = 5.0 # SHAPE OF DISTRIBUTION
       f_X = np.zeros(len(x))
       for i in range(0,len(x)):
           f_X[i] = (k/lamda)*((x[i]/lamda)**(k-1))* np.exp(-(x[i]/lamda)**k)

       params = [
        Parameter(order=1, distribution='Weibull', shape_parameter_A=lamda, shape_parameter_B=k),
        Parameter(order=1, distribution='Weibull', scale=lamda, shape=k),
        Parameter(order=1, distribution='Weibull', lamda=lamda, k=k),
        Weibull(order=1, shape_parameter_A=lamda, shape_parameter_B=k),
        Weibull(order=1, scale=lamda, shape=k),
        Weibull(order=1, lamda=lamda, k=k)
       ]
       for param in params:
        myBasis = Basis('univariate')
        myPoly = Poly(param, myBasis, method='numerical-integration')
        myPoly.set_model(blackbox)
        mean, variance = myPoly.get_mean_and_variance()
        xi_o = np.random.rand(N,1)
        xi = lamda * (-np.log(xi_o))**(1.0/k)
        yi = evaluate_model(np.reshape(xi, (N, 1) ), blackbox)
        samples = param.get_samples(1000)
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

      params = [
        Parameter(order=1, distribution='truncated-gaussian', lower=a, upper=b, shape_parameter_A=shape_A, shape_parameter_B=shape_B),
        Parameter(order=1, distribution='truncated-gaussian', low=a, up=b, mean=shape_A, variance=shape_B),
        Parameter(order=1, distribution='truncated-gaussian', lower=a, upper=b, mu=shape_A, var=shape_B),
        TruncatedGaussian(order=1, lower=a, upper=b, shape_parameter_A=shape_A, shape_parameter_B=shape_B),
        TruncatedGaussian(order=1, low=a, up=b, mean=shape_A, variance=shape_B),
        TruncatedGaussian(order=1, lower=a, upper=b, mu=shape_A, var=shape_B)
      ]
      for param in params:
        myBasis = Basis('univariate')
        myPoly = Poly(param, myBasis, method='numerical-integration')
        myPoly.set_model(blackbox)
        mean, variance = myPoly.get_mean_and_variance()
        a = x
        b = param.get_pdf(a) # analytical!
        samples = param.get_samples(3000)
        yi = samples
        eq_m = float('%.4f' %mean)
        mc_m = float('%.4f' %np.mean(yi))
        error_mean = np.testing.assert_almost_equal(eq_m, param.mean, decimal=1, err_msg="difference greater than imposed tolerance for mean value")
        eq_v = float('%.4f' %variance)
        mc_v = float('%.4f' %np.var(yi))
        error_var = np.testing.assert_almost_equal(eq_v, param.variance, decimal=1, err_msg="difference greater than imposed tolerance for variance value")
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
      
      params = [
        Parameter(order=1, distribution='Chebyshev', lower=0.001, upper=0.99),
        Parameter(order=1, distribution='Chebyshev', low=0.001, up=0.99),
        Chebyshev(order=1, lower=0.001, upper=0.99),
        Chebyshev(order=1, low=0.001, up=0.99)
      ]
      for param in params:
        myBasis = Basis('univariate')
        myPoly = Poly(param, myBasis, method='numerical-integration')
        myPoly.set_model(blackbox)
        mean, variance = myPoly.get_mean_and_variance()
        a = x
        b = param.get_pdf(a) # analytical!
        samples = param.get_samples(3000)
        yi = samples
        eq_m = float('%.4f' %mean)
        mc_m = float('%.4f' %np.mean(yi))
        error_mean = np.testing.assert_almost_equal(eq_m, param.mean, decimal=1, err_msg="difference greater than imposed tolerance for mean value")
        eq_v = float('%.4f' %variance)
        mc_v = float('%.4f' %np.var(yi))
        error_var = np.testing.assert_almost_equal(eq_v, param.variance, decimal=1, err_msg="difference greater than imposed tolerance for variance value")
    def test_cauchy(self):
      x0 = 0.0
      gamma = 0.5
      x = np.linspace(-15.,15.,100)
      f_X = np.zeros(len(x))
      for i in range(0,len(x)):
        f_X[i] = 1.0/(np.pi*gamma*(1+(((x[i]-x0)/gamma)**2)))

      params = [
        Parameter(order=1, distribution='Cauchy', shape_parameter_A=x0, shape_parameter_B=gamma),
        Parameter(order=1, distribution='Cauchy', location=x0, scale=gamma),
        Parameter(order=1, distribution='Cauchy', loc=x0, scale=gamma),
        Cauchy(order=1, shape_parameter_A=x0, shape_parameter_B=gamma),
        Cauchy(order=1, location=x0, scale=gamma),
        Cauchy(order=1, loc=x0, scale=gamma)
      ]
      for param in params:
        myBasis = Basis('univariate')
        myPoly = Poly(param, myBasis, method='numerical-integration')
        myPoly.set_model(blackbox)
        mean, variance = myPoly.get_mean_and_variance()
        a = x
        b = param.get_pdf(a)
        samples = param.get_samples(1000)
        yi = samples
    def test_exponential(self):
      params = [
        Parameter(order=1, distribution='exponential', shape_parameter_A=2.0),
        Parameter(order=1, distribution='exponential', rate=2.0),
        Parameter(order=1, distribution='exponential', lamda=2.0),
        Exponential(order=1, shape_parameter_A=2.0),
        Exponential(order=1, rate=2.0),
        Exponential(order=1, lamda=2.0)
      ]
      for param in params:
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
      params = [
        Parameter(order=1, distribution='chi-squared', shape_parameter_A=2.0),
        Parameter(order=1, distribution='chi-squared', dofs=2.0),
        Parameter(order=1, distribution='chi-squared', k=2.0),
        Chisquared(order=1, shape_parameter_A=2.0),
        Chisquared(order=1, dofs=2.0),
        Chisquared(order=1, k=2.0)
      ]
      for param in params:
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
      params = [
        Parameter(order=1, distribution='chi', shape_parameter_A=2.0),
        Parameter(order=1, distribution='chi', dofs=2.0),
        Parameter(order=1, distribution='chi', k=2.0),
        Chi(order=1, shape_parameter_A=2.0),
        Chi(order=1, dofs=2.0),
        Chi(order=1, k=2.0)
      ]
      for param in params:
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
      params = [
        Parameter(order=1, distribution='pareto', shape_parameter_A=2.0, shape_parameter_B=1.0),
        Parameter(order=1, distribution='pareto', shape=2.0, scale=1.0),
        Parameter(order=1, distribution='pareto', alpha=2.0, xm=1.0),
        Pareto(order=1, shape_parameter_A=2.0, shape_parameter_B=1.0),
        Pareto(order=1, shape=2.0, scale=1.0),
        Pareto(order=1, alpha=2.0, xm=1.0)
      ]
      for param in params:
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
      params = [
        Parameter(order=1, distribution='gumbel', shape_parameter_A=2.0, shape_parameter_B=3.2),
        Parameter(order=1, distribution='gumbel', location=2.0, scale=3.2),
        Parameter(order=1, distribution='gumbel', loc=2.0, scale=3.2),
        Parameter(order=1, distribution='gumbel', mu=2.0, beta=3.2),
        Gumbel(order=1, shape_parameter_A=2.0, shape_parameter_B=3.2),
        Gumbel(order=1, location=2.0, scale=3.2),
        Gumbel(order=1, loc=2.0, scale=3.2),
        Gumbel(order=1, mu=2.0, beta=3.2)
      ]
      for param in params:
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
      params = [
        Parameter(order=1, distribution='logistic', shape_parameter_A=3.5, shape_parameter_B=2.2),
        Parameter(order=1, distribution='logistic', location=3.5, scale=2.2),
        Parameter(order=1, distribution='logistic', loc=3.5, s=2.2),
        Parameter(order=1, distribution='logistic', mu=3.5, s=2.2),
        Logistic(order=1, shape_parameter_A=3.5, shape_parameter_B=2.2),
        Logistic(order=1, location=3.5, scale=2.2),
        Logistic(order=1, loc=3.5, s=2.2),
        Logistic(order=1, mu=3.5, s=2.2)
      ]
      for param in params:
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
    def test_studentt(self):
      params = [
        Parameter(order=1, distribution='t', shape_parameter_A=3.5),
        Parameter(order=1, distribution='t', dofs=3.5),
        Parameter(order=1, distribution='t', nu=3.5),
        Studentst(order=1, distribution='t', shape_parameter_A=3.5),
        Studentst(order=1, distribution='t', dofs=3.5),
        Studentst(order=1, distribution='t', nu=3.5)
      ]
      for param in params:
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
    def test_lognormal(self):
      params = [
        Parameter(order=1, distribution='lognormal', shape_parameter_A=0.0, shape_parameter_B=0.25),
        Parameter(order=1, distribution='lognormal', mean=0.0, standard_deviation=0.25),
        Parameter(order=1, distribution='lognormal', mu=0.0, std=0.25),
        Parameter(order=1, distribution='lognormal', mu=0.0, sigma=0.25),
        Lognormal(order=1, shape_parameter_A=0.0, shape_parameter_B=0.25),
        Lognormal(order=1, mean=0.0, standard_deviation=0.25),
        Lognormal(order=1, mu=0.0, std=0.25),
        Lognormal(order=1, mu=0.0, sigma=0.25)
      ]
      for param in params:
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
        np.testing.assert_almost_equal(variance, param.variance, decimal=1)
    def test_rayleigh(self):
      params = [
        Parameter(order=1, distribution='rayleigh', shape_parameter_A=2.0),
        Parameter(order=1, distribution='rayleigh', scale=2.0),
        Parameter(order=1, distribution='rayleigh', sigma=2.0),
        Rayleigh(order=1, shape_parameter_A=2.0),
        Rayleigh(order=1, scale=2.0),
        Rayleigh(order=1, sigma=2.0)
      ]
      for param in params:
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
    def test_uniform(self):
      params = [
        Parameter(order=5, distribution='uniform', lower=-1., upper=15.),
        Parameter(order=5, distribution='uniform', low=-1., up=15.),
        Parameter(order=5, distribution='uniform', bottom=-1., top=15.),
        Uniform(order=5, lower=-1., upper=15.),
        Uniform(order=5, low=-1., up=15.),
        Uniform(order=5, bottom=-1., top=15.),
      ]
      for param in params:
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
    def test_triangular(self):
      params = [
        Parameter(order=5, distribution='triangular', lower=-1., upper=15., shape_parameter_A=5),
        Parameter(order=5, distribution='triangular', a=-1., b=15., c=5),
        Parameter(order=5, distribution='triangular', a=-1., b=15., mode=5),
        Triangular(order=5, lower=-1., upper=15., shape_parameter_A=5),
        Triangular(order=5, a=-1., b=15., c=5),
        Triangular(order=5, a=-1., b=15., mode=5)
      ]
      for param in params:
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
      mu = 3.0
      sigma_2 = 0.5
      pdf_function = Weight(lambda x: 1./np.sqrt(2. * np.pi * sigma_2) * np.exp(-0.5 * (x - mu)**2/sigma_2 ), support=[-12., 12.] )
      param = Parameter(order=10, distribution='analytical', weight_function=pdf_function)
      s_values, pdf = param.get_pdf()
      s_values, cdf = param.get_cdf()
      s_samples = param.get_samples(6000)
      param.get_description()
      s_samples = param.get_icdf(np.linspace(0., 1., 30))
      myBasis = Basis('univariate')
      myPoly = Poly(param, myBasis, method='numerical-integration')
      myPoly.set_model(blackbox)
      mean, variance = myPoly.get_mean_and_variance()
      np.testing.assert_almost_equal(mean, paramtest.shape_parameter_A, decimal=2)
      np.testing.assert_almost_equal(variance, paramtest.shape_parameter_B, decimal=2)
    def test_custom2(self):
      a = 3.
      b = 6.
      c = 4.
      mean = (a + b + c)/3.
      var = (a**2 + b**2 + c**2 - a*b - a*c - b*c)/18.
      pdf = Weight(lambda x : 2*(x-a)/((b-a)*(c-a)) if (a <= x < c) \
                              else( 2/(b-a) if (x == c) \
                              else( 2*(b-x)/((b-a)*(b-c)))), \
                  support=[a, b], pdf=True)
      np.testing.assert_almost_equal(mean, pdf.mean, decimal=5)
      np.testing.assert_almost_equal(var, pdf.variance, decimal=5)
      s = Parameter(distribution='analytical', weight_function=pdf, order=2)
      s_samples = s.get_samples(50000)
      basis = Basis('univariate')
      poly = Poly(s, basis, method='numerical-integration')
      def model(input):
        return input**2
      poly.set_model(model)
      feval = evaluate_model(s_samples, model)
      mean2, variance2 = poly.get_mean_and_variance()
      np.testing.assert_almost_equal(mean2/100., np.mean(feval)/100., decimal=2)
if __name__ == '__main__':
    unittest.main()
