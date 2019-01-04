from unittest import TestCase
import unittest
from equadratures import *
import numpy as np

def piston(x):
    mass, area, volume, spring, pressure, ambtemp, gastemp = x[0], x[1], x[2], x[3], x[4], x[5], x[6]
    A = pressure * area + 19.62*mass - (spring * volume)/(1.0 * area)
    V = (area/(2*spring)) * ( np.sqrt(A**2 + 4*spring * pressure * volume * ambtemp/gastemp) - A)
    C = 2 * np.pi * np.sqrt(mass/(spring + area**2 * pressure * volume * ambtemp/(gastemp * V**2)))
    return C

def G_fun(x):
    f = 1.0
    for i in range(4):
        t = (np.abs(4*x[i] - 2) + i**2.0) * 1.0/(1 + i**2.0)
        f = f * t
    return f

class TestStats(TestCase):

    def test_sobol(self):
        order_parameters = 3
        mass = Parameter(distribution='uniform', lower=30.0, upper=60.0, order=order_parameters)
        area = Parameter(distribution='uniform', lower=0.005, upper=0.020, order=order_parameters)
        volume = Parameter(distribution='uniform', lower=0.002, upper=0.010, order=order_parameters)
        spring = Parameter(distribution='uniform', lower=1000., upper=5000., order=order_parameters)
        pressure = Parameter(distribution='uniform', lower=90000., upper=110000., order=order_parameters)
        ambtemp = Parameter(distribution='uniform', lower=290., upper=296., order=order_parameters)
        gastemp = Parameter(distribution='uniform', lower=340., upper=360., order=order_parameters)
        parameters = [mass, area, volume, spring, pressure, ambtemp, gastemp]

        mybasis = Basis('Total order')
        Pleastsquares = Polylsq(parameters, mybasis, mesh='tensor', optimization='greedy-qr', oversampling=1.0)
        Pleastsquares.computeCoefficients(piston)
        Sleastsquares = Pleastsquares.getStatistics()

        data = Sleastsquares.getSobol(1).values()
        for i in range(0, len(parameters)):
            print( float(data[i]) * 10**2 * Sleastsquares.variance )

        sobol_info = Sleastsquares.getSobol(2)
        for key, value in sobol_info.iteritems():
            print( str('Parameter numbers: ')+str(key)+', Sobol index value: '+str(value) )
    
    def test_higher_order(self):
        degree = 5
        x0 = Parameter(distribution="Uniform", lower=0.0, upper=1.0, order=degree)
        x1 = Parameter(distribution="Uniform", lower=0.0, upper=1.0, order=degree)
        x2 = Parameter(distribution="Uniform", lower=0.0, upper=1.0, order=degree)
        x3 = Parameter(distribution="Uniform", lower=0.0, upper=1.0, order=degree)
        parameters = [x0,x1,x2,x3]

        basis = Basis('Tensor grid')
        uqProblem = Polyint(parameters,basis)
        uqProblem.computeCoefficients(G_fun)
        stats = uqProblem.getStatistics()

        np.testing.assert_almost_equal(stats.mean, 1.03619468893, decimal=6, err_msg = "Difference greated than imposed tolerance for mean value")
        np.testing.assert_almost_equal(stats.variance, 0.423001291441, decimal=6, err_msg = "Difference greated than imposed tolerance for variance value")
        np.testing.assert_almost_equal(stats.skewness, 0.874198787521, decimal=6, err_msg = "Difference greated than imposed tolerance for skewness value")
        np.testing.assert_almost_equal(stats.kurtosis, 3.03775388049, decimal=6, err_msg = "Difference greated than imposed tolerance for kurtosis value")
        
        s1 = stats.getCondSkewness(1)
        s2 = stats.getCondSkewness(2)
        #k1 = stats.getCondKurtosis(1)
        #k2 = stats.getCondKurtosis(2)

        #print sum(v1.values()) + sum(v2.values())
        print( sum(s1.values()) + sum(s2.values()) )
        #print sum(k1.values()) + sum(k2.values())


if __name__== '__main__':
    unittest.main()
