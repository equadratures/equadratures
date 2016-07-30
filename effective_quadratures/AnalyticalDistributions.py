#!/usr/bin/env python
import numpy as np
from scipy.special import erf
"""

   Analytical definitions for some sample PDFs. Functions in this file are
   called by PolyParams when constructing "custom" orthogonal polynomials,
   which require Stieltejes procedure for computing the recurrence coefficients.

    Pranay Seshadri
    ps583@cam.ac.uk

    Copyright (c) 2016 by Pranay Seshadri
"""
def Gaussian(N, mu, sigma):
  x = np.linspace(-15*sigma, 15*sigma, N) # x scaled by the standard deviation!
  w = 1.0/( np.sqrt(2 * sigma**2 * np.pi) * np.exp(-(x - mu)**2 * 1.0/(2 * sigma**2) )
  w = w/np.sum(w) # normalize!
  return x, w
  
def truncatedGaussian(N, mu, sigma, a, b):


def GaussianPDF():

def GaussianCDF():


