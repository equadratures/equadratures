#!/usr/bin/env python
from unittest import TestCase
import unittest
from equadratures import *
import numpy as np

class TestParameter(TestCase):

    def test_declarations(self):

        # Parameter test 1: getPDFs()
        var1 = Parameter(points=12, shape_parameter_A=2, shape_parameter_B=3, param_type='TruncatedGaussian', lower=3, upper=10)
        x, y = var1.getPDF(50)

        # Parameter test 2: getRecurrenceCoefficients()
        var2 = Parameter(points=15, param_type='Uniform', lower=-1, upper=1)
        ab = var2.getRecurrenceCoefficients()

        # Parameter test 3: getJacobiMatrix()
        var3 = Parameter(points=5, param_type='Beta', lower=0, upper=5, shape_parameter_A=2, shape_parameter_B=3)
        J = var3.getJacobiMatrix()

        # Parameter test 4: getJacobiEigenvectors()
        var4 = Parameter(points=5, param_type='Gaussian', shape_parameter_A=0, shape_parameter_B=2)
        V = var4.getJacobiEigenvectors()

        # Parameter test 5: computeMean()
        var5 = Parameter(points=10, param_type='Weibull', shape_parameter_A=1, shape_parameter_B=5)
        mu = var5.computeMean()

        # Parameter test 6: getOrthoPoly(points):
        x = np.linspace(-1,1,15)
        var6 = Parameter(points=10, param_type='Uniform', lower=-1, upper=1)
        poly = var6.getOrthoPoly(x)

        # Parameter test 7: Now with derivatives
        var7 = Parameter(points=7, param_type='Uniform', lower=-1, upper=1, derivative_flag=1)
        poly, derivatives = var7.getOrthoPoly(x)

        # Parameter test 8: getLocalQuadrature():
        var8 = Parameter(points=5, shape_parameter_A=0.8, param_type='Exponential')
        p, w = var8.getLocalQuadrature()
        return 0

if __name__ == '__main__':
    unittest.main()
