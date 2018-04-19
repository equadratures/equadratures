from unittest import TestCase
import unittest
from equadratures import *
import numpy as np


class TestPolynomials(TestCase):

    def test_chebyshev(self):
        # Experiment 1
        p1 = Parameter(param_type='Chebyshev', lower=-3.0, upper=1.0, order=4)
        xx = np.linspace(0.0, 1.0, 200)
        P , grad_P = p1._getOrthoPoly(xx)
        polynomialplot(P, xx)

if __name__ == '__main__':
    unittest.main()
