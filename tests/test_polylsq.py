#!/usr/bin/env python
from unittest import TestCase
import unittest
from equadratures import *
import numpy as np

class TestPolylsq(TestCase):
    
    def fun(x):
        return 1.0/(1 + 50*(x[0]- 0.9)**2 + 50*(x[1] + 0.9)**2 )
    
    
    no_of_subsamples = 10
    x1 = Parameter(param_type="Uniform", lower=-1, upper=1, points=no_of_subsamples)
    x2 = Parameter(param_type="Uniform", lower=-1, upper=1, points=no_of_subsamples)
    parameters = [x1, x2]
    Hyperbolic = IndexSet("Hyperbolic basis", orders=[no_of_subsamples-1,no_of_subsamples-1], q=0.3)
    esq = Polylsq(parameters, Hyperbolic)
    minimum_subsamples = esq.least_no_of_subsamples_reqd() 
    esq.set_no_of_evals(minimum_subsamples)
    psmall = esq.subsampled_quadrature_points
    xvec = np.linspace(-1.,1.,40)                               
    x,y = np.meshgrid(xvec, xvec)
    z =  1.0/(1 + 50*(x - 0.9)**2 + 50*(y + 0.9)**2 )       
    stackOfPoints, x1, x2 = meshgrid(-1.0, 1.0, 40, 40)
    zapprox = esq.getPolynomialApproximation(stackOfPoints, fun)


if __name__ == '__main__':
    unittest.main()
