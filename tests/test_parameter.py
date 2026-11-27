from unittest import TestCase
import unittest
from equadratures import *
import numpy as np

class TestParameter(TestCase):

    def test_param_basic(self):
        myparameter=Parameter(lower=-1., upper=1.)
        np.testing.assert_equal(myparameter.variable, 'parameter')
        np.testing.assert_equal(myparameter.name, 'uniform')
        np.testing.assert_equal(myparameter.order, 1)
        myparameter2 = Parameter(lower=250., upper=300., variable='horsepower')
        np.testing.assert_equal(myparameter2.variable, 'horsepower')

if __name__== '__main__':
    unittest.main()