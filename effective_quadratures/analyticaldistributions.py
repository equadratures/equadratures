#!/usr/bin/env python
"""Sample probability density functions"""
import numpy as np
from scipy.special import erf, erfinv, gamma, beta, betainc, gammainc

# inverse-CDFs
def iCDF_Gaussian(xx, mu, sigma):
    return mu + sigma * np.sqrt(2.0) * erfinv(2.0*xx - 1.0)

def iCDF_CauchyDistribution(xx, x0, gammavalue):
    return x0 + gamma * np.tan(np.pi * (xx - 0.5))

def iCDF_WeibullDistribution(xx, lambda_value, k):
    return lambda_value * (-np.log(1.0 - xx))**(1.0/k) 

def iCDF_ExponentialDistribution(xx, lambda_value):
    return (-np.log(1.0 - xx))/(lambda_value)

def iCDF_BetaDistribution(xx, a, b, lower, upper):
    # Insert code for computing an inverse beta CDF here!
     [x,y]=BetaDistribution(1000,a,b,lower,upper)
     c=[]
     c.append(0)
     for i in range(1, len(x)):
        c.append(c[i-1]+(x[i]-x[i-1])*(y[i]+y[i-1])*.5)
     
     for i in range(1, len(x)):
        c[i]=c[i]/c[len(x)-1]
 
     for i in range(0, len(x)-1):
        if ( (xx>=c[i]) and (xx<=c[i+1]) ): 
           yy=(xx-c[i])/(c[i+1]-c[i])*(x[i+1]-x[i])+x[i]
           break
     return xx

def iCDF_TruncatedGaussian(xx, mu, sigma, a, b):
    yy = iCDF_Gaussian(xx, mu, sigma)
    for i in range(0, len(xx)):
        if(yy[i,0] < a or yy[i,0] > b):
            yy[i,0] = 0
    return yy

# Cumulative distribution
def CDF_GaussianDistribution(N, mu, sigma):
    x = np.linspace(-15*sigma, 15*sigma, N)
    x = x + mu # scaling it by the mean!
    w = 0.5*(1 + erf((x - mu)/(sigma * np.sqrt(2) ) ) )
    return x, w

def CDF_TruncatedGaussianDistribution(N, mu, sigma, a, b):
    def cumulative(x):
        return 0.5 * (1 + erf(x/np.sqrt(2)))
    x = np.linspace(a, b, N) 
    zeta = (x - mu)/(sigma)
    alpha = (a - mu)/(sigma)
    beta = (b - mu)/(sigma)
    Z = cumulative(beta) - cumulative(alpha)
    w = (cumulative(zeta) - cumulative(alpha))/(Z)
    return x, w

def CDF_BetaDistribution(N, a, b, lower, upper):
    x = np.linspace(0, 1, N)
    w = np.zeros((N,1))
    for i in range(0, N):
        w[i] = betainc(a, b, x[i])
    return x, w

def CDF_WeibullDistribution(N, lambda_value, k):
    x = np.linspace(0, 15/k, N)
    w = 1 - np.exp(-1.0 * ( (x) / (lambda_value * 1.0)  )**k )
    return x, w

def CDF_UniformDistribution(N, lower, upper):
     x = np.linspace(lower, upper, N)
     w = np.zeros((N, 1))
     for i in range(0, N):
         w[i] = (x[i] - lower)/(upper - lower)
     return x, w

def CDF_GammaDistribution(N, k, theta):
    x = np.linspace(0, k*theta*10, N)
    w = 1.0/(gamma(k)) * gammainc(k, x/theta)
    return x, w

def CDF_CauchyDistribution(N, x0, gammavalue):
    x = np.linspace(-15*gammavalue, 15*gammavalue, N)
    x = x + x0
    w = 1.0/np.pi * np.arctan((x - x0) / gammavalue) + 0.5
    return x, w

def CDF_ExponentialDistribution(N, lambda_value):
    x = np.linspace(0, 20*lambda_value, N)
    w = 1 - np.exp(-lambda_value * x)
    return x, w

# PDFs
def PDF_UniformDistribution(N, lower, upper):
    x = np.linspace(lower, upper, N)
    w = 0*x + (1.0)/(upper - lower)
    return x, w

def PDF_BetaDistribution(N, a, b, lower, upper):
    x = np.linspace(0, 1, N)
    w = (x**(a - 1) * (1 - x)**(b - 1))/(beta(a, b) )
    xreal = np.linspace(lower, upper, N)
    wreal = w * (1.0)/(upper - lower)
    return xreal, wreal

def PDF_GaussianDistribution(N, mu, sigma):
    x = np.linspace(-15*sigma, 15*sigma, N)
    x = x + mu # scaling it by the mean!
    w = 1.0/( np.sqrt(2 * sigma**2 * np.pi) ) * np.exp(-(x - mu)**2 * 1.0/(2 * sigma**2) )
    return x, w

def PDF_WeibullDistribution(N, lambda_value, k):
    x = np.linspace(0, 15/k, N)
    w = k/lambda_value * (x/lambda_value)**(k-1) * np.exp(-1.0 * (x/lambda_value)**k )
    return x, w

def PDF_GammaDistribution(N, k, theta):
    x = np.linspace(0, k*theta*10, N)
    w = 1.0/(gamma(k) * theta**k) * x**(k-1) * np.exp(-x/theta)
    return x, w

def PDF_CauchyDistribution(N, x0, gammavalue):
    x = np.linspace(-15*gammavalue, 15*gammavalue, N)
    x = x + x0
    w = 1.0/(pi * gammavalue * (1 + ((x - x0)/(gammavalue))**2) )
    return x, w

def PDF_ExponentialDistribution(N, lambda_value):
    x = np.linspace(0, 20*lambda_value, N)
    w = lambda_value * np.exp(-lambda_value * x)
    return x, w

def  PDF_TruncatedGaussian(N, mu, sigma, a, b):
    x = np.linspace(a, b, N)
    w = 1.0/( np.sqrt(2 * sigma**2 * np.pi)) * np.exp(-(x - mu)**2 * 1.0/(2 * sigma**2) )
    w = 1.0/sigma * w
    first_term = GaussianCDF(b, mu, sigma)
    second_term = GaussianCDF(a, mu, sigma)
    w = w / (first_term - second_term)
    return x, w

def GaussianCDF(constant, mu, sigma):
    w = 1.0/2 * (1 + erf((constant - mu)/(sigma * np.sqrt(2))) )
    return w