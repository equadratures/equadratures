from unittest import TestCase
import unittest
from equadratures import *
import numpy as np
import math
class TestSamplingGeneration(TestCase):
    def test_sampling(self):
        d=5
        param = Parameter(distribution='uniform', order=3, lower= -1.0, upper=1.0)
        myparameters = [param for _ in range(d)]
        mybasis = Basis('total-order')
        mypoly1 = Poly(myparameters, mybasis, method='least-squares', sampling_args={'mesh':'tensor-grid', 'subsampling-algorithm':'qr', 'sampling-ratio':1.0} )
        mybasis2 = Basis('total-order')
        mypoly1 = Poly(myparameters, mybasis2, method='least-squares', sampling_args={'mesh':'induced', 'subsampling-algorithm':'qr', 'sampling-ratio':1.0} )
if __name__== '__main__':
    unittest.main()