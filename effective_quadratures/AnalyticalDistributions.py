#!/usr/bin/env python
import numpy as np
from scipy.special import erf
from scipy.special import gamma
"""

   Analytical definitions for some sample PDFs. Functions in this file are
   called by PolyParams when constructing "custom" orthogonal polynomials,
   which require Stieltejes procedure for computing the recurrence coefficients.
   Note that these definitions are not normalized -- and they are normalized when
   used in PolyParams.py


    Pranay Seshadri
    ps583@cam.ac.uk

    Copyright (c) 2016 by Pranay Seshadri
"""

def Gaussian(mu, sigma, N):
  x, w = GaussianPDF(mu, sigma, N)
  return x, w

def WeibullDistribution(N, lambda_value, k):
    x = np.linspace(0, 15/k, N)
    w = k/lambda_value * (x/lambda_value)**(k-1) * np.exp(-1.0 * (x/lambda_value)**k )
    return x, w

def GammaDistribution(N, k, theta):
    x = np.linspace(0, k*theta*10, N)
    w = 1.0/(gamma(k) * theta**k) * x**(k-1) * np.exp(-x/theta)
    return x, w

def CauchyDistribution(N, x0, gammavalue):
    x = np.linspace(-15*gammavalue, 15*gammavalue, N)
    x = x + x0
    w = 1.0/(pi * gammavalue * (1 + ((x - x0)/(gammavalue))**2) )
    return x, w

def ExponentialDistribution(N, lambda_value):
    x = np.linspace(0, 15*lambda_value, N)
    w = lambda_value * np.exp(-lambda_value * x)
    return x, w

def TruncatedGaussian(N, mu, sigma, a, b):
    x = np.linspace(a, b, N)
    w = 1.0/( np.sqrt(2 * sigma**2 * np.pi)) * np.exp(-(x - mu)**2 * 1.0/(2 * sigma**2) )
    w = 1.0/sigma * w
    first_term = GaussianCDF(b, mu, sigma)
    second_term = GaussianCDF(a, mu, sigma)
    w = w / (first_term - second_term)
    return x, w

def GaussianPDF(mu, sigma, N):
    x = np.linspace(-15*sigma, 15*sigma, N)
    x = x + mu # scaling it by the mean!
    w = 1.0/( np.sqrt(2 * sigma**2 * np.pi) ) * np.exp(-(x - mu)**2 * 1.0/(2 * sigma**2) )
    return x, w

def GaussianCDF(constant, mu, sigma):
    w = 1.0/2 * (1 + erf((constant - mu)/(sigma * np.sqrt(2))) )
    return w
