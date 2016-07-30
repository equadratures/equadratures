#!/usr/bin/env python
import numpy as np
from scipy.special import erf
"""

   Analytical definitions for some sample PDFs. Functions in this file are
   called by PolyParams when constructing "custom" orthogonal polynomials,
   which require Stieltejes procedure for computing the recurrence coefficients.
   Note that these definitions are not normalized -- and they are normalized when
   used in PolyParams.py


    Q. Should I just set the mean to be zero??

    Pranay Seshadri
    ps583@cam.ac.uk

    Copyright (c) 2016 by Pranay Seshadri
"""
def Gaussian(mu, sigma, N):
  x, w = GaussianPDF(mu, sigma, N)
  return x, w

#def LogNormalDistribution(N, mu, sigma):
#    x = np.linspace(0, )
#
# def GammaDistribution()
#
# def CauchyDistribution()
#

def ExponentialDistribution(N, lambda_value):
    x = np.linspace(0, 15*lambda_value, N)
    w = lambda_value * np.exp(-lambda_value * x)
    return x, w

def TruncatedGaussian(N, mu, sigma, a, b):
    x = np.linspace(a, b, N)
    w = 1.0/( np.sqrt(2 * sigma**2 * np.pi)) * np.exp(-(x - mu)**2 * 1.0/(2 * sigma**2) )
    w = 1.0/sigma * w
    first_term = GaussianCDF(b, mean, sigma)
    second_term = GaussianCDF(a, mean, sigma)
    w = w / (first_term - second_term)
    return x, w

def GaussianPDF(mu, sigma, N):
    x = np.linspace(-10*sigma, 10*sigma, N)
    x = x + mu # scaling it by the mean!
    w = 1.0/( np.sqrt(2 * sigma**2 * np.pi) ) * np.exp(-(x - mu)**2 * 1.0/(2 * sigma**2) )
    return x, w

def GaussianCDF(constant, mean, sigma):
    w = 1.0/2 * (1 + erf((constant - mu)/(sigma * np.sqrt(2))) )
    return w
