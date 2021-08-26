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

    def test_param_data_uniform(self):
        lower=5
        upper=10
        ScipyParam=stats.uniform(loc=lower,scale=upper-lower)
        myparam=Parameter(lower=lower,upper=upper)
        arr=ScipyParam.rvs(size=1000,random_state=2021)
        myparam2=Parameter(data=arr)

        np.testing.assert_almost_equal(myparam.mean,myparam2.mean,decimal=3,err_msg ="Difference_{}")
        np.testing.assert_almost_equal(myparam.variance, myparam2.variance, decimal=3, err_msg="Difference_{}")

    def test_param_data_expon(self):
        shape_parameter_A=5
        ScipyParam=stats.expon(scale=1/shape_parameter_A,loc=0)
        myparam=Parameter(distribution='exponential',shape_parameter_A=shape_parameter_A)
        arr=ScipyParam.rvs(size=1000,random_state=2021)
        myparam2=Parameter(distribution='exponential',data=arr)

        np.testing.assert_almost_equal(myparam.mean,myparam2.mean,decimal=3,err_msg ="Difference_{}")
        np.testing.assert_almost_equal(myparam.variance, myparam2.variance, decimal=3, err_msg="Difference_{}")

    def test_param_data_lognorm(self):
        shape_parameter=0.15
        ScipyParam=stats.lognorm(s=shape_parameter,loc=0)
        myparam=Parameter(distribution='lognormal',shape_parameter_A=shape_parameter)
        arr=ScipyParam.rvs(size=1000,random_state=2021)
        myparam2=Parameter(distribution='lognormal',data=arr)

        np.testing.assert_almost_equal(myparam.mean,myparam2.mean,decimal=2,err_msg ="Difference_{}")
        np.testing.assert_almost_equal(myparam.variance, myparam2.variance, decimal=2, err_msg="Difference_{}")

    def test_param_data_gauss(self):
        shape_parameter_A=5
        shape_parameter_B=10
        ScipyParam=stats.norm(loc=shape_parameter_A,scale=np.sqrt(shape_parameter_B))
        myparam=Parameter(distribution='gaussian',shape_parameter_A=shape_parameter_A,shape_parameter_B=shape_parameter_B)
        arr=ScipyParam.rvs(size=1000,random_state=2021)
        myparam2=Parameter(distribution='gaussian',data=arr)

        np.testing.assert_almost_equal(myparam.mean,myparam2.mean,decimal=1,err_msg ="Difference_{}")
        np.testing.assert_almost_equal(myparam.variance, myparam2.variance, decimal=1, err_msg="Difference_{}")

if __name__== '__main__':
    unittest.main()