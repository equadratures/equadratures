from unittest import TestCase
import unittest
from equadratures import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf, gamma

class Test_Quadratures(TestCase): 

    def test_tensor_grid(self):
        p1 = Parameter(distribution='uniform', lower=-1., upper=1., order=3)
        myBasis = Basis('Tensor grid')
        Pols = Polyint([p1, p1], myBasis)
        print Pols.quadraturePoints, Pols.quadratureWeights


if __name__ == '__main__':
    unittest.main()
