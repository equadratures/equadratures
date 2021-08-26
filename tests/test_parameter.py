from unittest import TestCase
import unittest

import equadratures
from equadratures import *
import numpy as np
import scipy.stats as stats

class TestParameter(TestCase):

    def test_param_basic(self):
        myparameter=Parameter(lower=-1., upper=1.)
        np.testing.assert_equal(myparameter.variable, 'parameter')
        np.testing.assert_equal(myparameter.name, 'Uniform')
        np.testing.assert_equal(myparameter.order, 1)
        myparameter2 = Parameter(lower=250., upper=300., variable='horsepower')
        np.testing.assert_equal(myparameter2.variable, 'horsepower')

    def test_param_data(self):
        lower=5
        upper=10
        ScipyParam=stats.uniform(loc=lower,scale=upper-lower)
        myparam=Parameter(lower=lower,upper=upper)
        arr=ScipyParam.rvs(size=1000,random_state=2021)
        myparam2=Parameter(data=arr)

        np.testing.assert_almost_equal(myparam.mean,myparam2.mean,decimal=3,err_msg ="Difference_{}")
        np.testing.assert_almost_equal(myparam.variance, myparam2.variance, decimal=3, err_msg="Difference_{}")

if __name__== '__main__':
    unittest.main()