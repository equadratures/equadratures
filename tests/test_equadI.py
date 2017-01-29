#!/usr/bin/env python
from unittest import TestCase
import unittest
from effective_quadratures.parameter import Parameter
from effective_quadratures.polynomial import Polynomial
from effective_quadratures.indexset import IndexSet
from effective_quadratures.effectivequads import EffectiveSubsampling
from effective_quadratures.computestats import Statistics
import numpy as np

class TestEquad(TestCase):
    
    def test_vegetation_problem(self):

        def fun(x):
            return 1.0/(1 + 50*(x[0]- 0.9)**2 + 50*(x[1] + 0.9)**2 )
        
        value_large = 10
        x1 = Parameter(param_type="Uniform", lower=-1, upper=1, points=value_large)
        x2 = Parameter(param_type="Uniform", lower=-1, upper=1, points=value_large)
        uq = Polynomial([x1,x2])
        p, w = uq.getPointsAndWeights()

        no_of_subsamples = 10
        x1 = Parameter(param_type="Uniform", lower=-1, upper=1, points=no_of_subsamples)
        x2 = Parameter(param_type="Uniform", lower=-1, upper=1, points=no_of_subsamples)
        parameters = [x1, x2]
        Hyperbolic = IndexSet("Hyperbolic basis", orders=[no_of_subsamples-1,no_of_subsamples-1], q=0.3)
        esq = EffectiveSubsampling(parameters, Hyperbolic)
        minimum_subsamples = esq.least_no_of_subsamples_reqd() 
        esq.set_no_of_evals(minimum_subsamples)

if __name__ == '__main__':
    unittest.main()