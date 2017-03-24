#!/usr/bin/env python
from unittest import TestCase
import unittest
from effective_quadratures.parameter import Parameter
import effective_quadratures.integrals as integrals
from effective_quadratures.indexset import IndexSet
from effective_quadratures.polynomial import Polynomial
import numpy as np

class TestPDFPlots(TestCase):
    
    def testinputPDFs(self):
        # Output a histogram based on 1000 samples 
        X = Parameter(points=3, shape_parameter_A=15, shape_parameter_B=2.5, param_type='Gaussian')
        X.getSamples(10000, graph=1)
        
    
    def testoutputPDFs(self):
        
        def expfun(x):
            return np.exp(x[0] + x[1] ) + 0.5*np.cos(x[0]*2*np.pi)

        # Compare actual function with polynomial approximation
        s1 = Parameter(lower=-1, upper=1, points=6, shape_parameter_A=0, shape_parameter_B=2.5, param_type='Gaussian')
        s2 = Parameter(lower=0, upper=5, points=6, shape_parameter_A=1.0, shape_parameter_B=3.0, param_type='Weibull')
        T = IndexSet('Tensor grid', [5,5])
        uq = Polynomial([s1,s2], T)
        output = uq.getPDF(expfun, graph=1)

if __name__ == '__main__':
    unittest.main()
