from unittest import TestCase
import unittest
from equadratures import *
import numpy as np
import scipy.stats as st

def fun(x):
    a = 1.0
    b = 100.0
    return (a - x[0])**2 + b * (x[1] - x[0]**2)**2

class TestC(TestCase):

    def test_quadratic(self):
        zeta_1 = Parameter(distribution='uniform', order=4, lower= -2.0, upper=2.0)
        zeta_2 = Parameter(distribution='uniform', order=4, lower=-1.0, upper=3.0)

        myBasis1 = Basis('tensor-grid')
        myBasis2 = Basis('total-order')
        myPoly1 = Poly([zeta_1, zeta_2], myBasis1, method='numerical-integration')
        myPoly2 = Poly([zeta_1, zeta_2], myBasis2, method='least-squares', \
                    args={'mesh':'tensor-grid', 'subsampling-algorithm':'qr', \
                            'sampling-ratio':1.0})
        myPoly1.set_model(fun)
        myPoly2.set_model(fun)

        pts1 = myPoly1.get_points()
        pts2 = myPoly2.get_points()




if __name__== '__main__':
    unittest.main()