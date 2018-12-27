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
            print float(data[i]) * 10**2 * Sleastsquares.variance

        sobol_info = Sleastsquares.getSobol(2)
        for key, value in sobol_info.iteritems():
            print str('Parameter numbers: ')+str(key)+', Sobol index value: '+str(value)
                                          
if __name__== '__main__':
    unittest.main()
