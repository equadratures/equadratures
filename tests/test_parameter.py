from unittest import TestCase
import unittest
from equadratures import *
import numpy as np

class TestParameter(TestCase):

    def test_param_basic(self):
        myparameter=Parameter(lower=-1., upper=1.)
        np.testing.assert_equal(myparameter.variable, 'parameter')
        np.testing.assert_equal(myparameter.name, 'Uniform')
        np.testing.assert_equal(myparameter.order, 1)
        myparameter2 = Parameter(lower=250., upper=300., variable='horsepower')
        np.testing.assert_equal(myparameter2.variable, 'horsepower')

    def test_param_data(self):
        arr=np.linspace(0,100,200)
        myparameter=Parameter(data=arr)
        np.testing.assert_equal(myparameter.variable, 'parameter')
        np.testing.assert_equal(myparameter.order, 1)
        myparameter2=Parameter(lower=min(arr),upper=max(arr))
        mybasis=Basis('total-order')
        myPoly=Poly(myparameter,mybasis,method='least-squares')
        myPoly2=Poly(myparameter2,mybasis,method='least-squares')

        np.testing.assert_allclose(myPoly.get_points_and_weights()[0],myPoly2.get_points_and_weights()[0],
                                   err_msg = "Difference greated than imposed tolerance")
        np.testing.assert_allclose(myPoly.get_points_and_weights()[1], myPoly2.get_points_and_weights()[1],
                                   err_msg="Difference greated than imposed tolerance")

if __name__== '__main__':
    unittest.main()