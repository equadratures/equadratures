from unittest import TestCase
import unittest
from equadratures import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf, gamma
N = 900000

def plot(x, f_X, a, b, samples, dist_name):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.hist(samples, 30, normed=1, facecolor='orangered', edgecolor='black', linewidth=0.5)
    plt.plot(x, f_X, c='navy', lw=3)
    plt.plot(a, b, '--', c='green')
    adjust_spines(ax, ['left', 'bottom'])
    plt.xlabel('Viscosity', fontsize=13)
    plt.ylabel('PDF', fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.title(dist_name)
    ax.set_xlabel('X variable')
    ax.set_ylabel('PDF')
    plt.tight_layout()
    plt.show()
def blackbox(x):
    return x

def test_truncated_gauss():
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
    a,b = xo.getPDF(N=150) # analytical!
    samples = xo.getSamples(m=3000)
    yi = samples
    plot(x, f_X, a, b, samples, xo.name)
    print xo.name, xo.shape_parameter_A, xo.shape_parameter_B, xo.lower, xo.upper
    print 'Effective quadrature mean: ', myStats.mean#, myStats.variance
    print 'Effective quadrature variance:' , myStats.variance
    print 'Analytical mean :', xo.mean 
    print 'Analytical variance :', xo.variance
    print '\n'
    eq_m = float('%.4f' %myStats.mean)
    mc_m = float('%.4f' %np.mean(yi))
    error_mean = np.testing.assert_almost_equal(eq_m, xo.mean, decimal=2, err_msg="difference greater than imposed tolerance for mean value")
    eq_v = float('%.4f' %myStats.variance)
    mc_v = float('%.4f' %np.var(yi))
    error_var = np.testing.assert_almost_equal(eq_v, xo.variance, decimal=1, err_msg="difference greater than imposed tolerance for variance value")

test_truncated_gauss()
