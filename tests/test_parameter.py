from unittest import TestCase
import unittest
from equadratures import *
import numpy as np

class TestParameters(TestCase):

    def test_parameters(self):
        # Experiment 1
        p1 = Parameter(param_type='Uniform', lower=-1.0, upper=5.5, order=3)
        p, w = p1._getLocalQuadrature()
        P , grad_P = p1._getOrthoPoly(p)

        p2 = Parameter(param_type='Uniform', lower=-1.0, upper=1.0, order=3)
        po, wo = p2._getLocalQuadrature()
        P2 , grad_P2 = p2._getOrthoPoly(po)

        x, y = p1.getPDF(50000)
        yy = p1.getSamples()
        histogram(yy, 'Samples', 'Histogram')

        x, y = p2.getPDF(50000)
        yy = p2.getSamples()
        histogram(yy, 'Samples', 'Histogram')
        del p, po, P, P2

        # Experiment 2
        p1 = Parameter(param_type='Beta', lower=-1.0, upper=5.5, shape_parameter_A = 3.0, shape_parameter_B = 2.0, order=3)
        p, w = p1._getLocalQuadrature()
        P , grad_P = p1._getOrthoPoly(p)

        p2 = Parameter(param_type='Beta', lower=0.0, upper=1.0, shape_parameter_A = 3.0, shape_parameter_B = 2.0, order=3)
        po, w = p2._getLocalQuadrature()
        P2 , grad_P2 = p2._getOrthoPoly(po)

        x, y = p1.getPDF(50000)
        lineplot(x, y, 'Samples', 'Kernel density estimate')

        x, y = p2.getPDF(50000)
        lineplot(x, y, 'Samples', 'Kernel density estimate')

    def test_custom_parameter(self):
        # Select a custom distribution!
        loc, scale = 0., 1.
        s = np.random.laplace(loc, scale, 5000)

        # Set up the parameter!
        p1 = Parameter(param_type='Custom', order=4, data=s)
        x, y = p1.getPDF(50000)
        yy = p1.getSamples()

        # Generate the plots!
        lineplot(x, y, 'Samples', 'Kernel density estimate')
        histogram(yy, 'Samples', 'Histogram')

    def test_chebyshev_parameter(self):
        # Set up the parameter!
        p1 = Parameter(param_type='Chebyshev', order=4, lower=0.0, upper=1.0)
        x, y = p1.getPDF(50000)
        yy = p1.getSamples()
        lineplot(x, y, 'Samples', 'Kernel density estimate')
        histogram(yy, 'Samples', 'Histogram')

if __name__ == '__main__':
    unittest.main()
