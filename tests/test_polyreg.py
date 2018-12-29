from unittest import TestCase
import unittest
from equadratures import *
import numpy as np

class TestPolyreg(TestCase):

    def test_classical1D(self):
        dimensions = 1
        M = 12
        param = Parameter(distribution='Uniform', lower=0, upper=1., order=1)
        myParameters = [param for i in range(dimensions)]
        x_train = np.mat([0,0.0714,0.1429,0.2857,0.3571,0.4286,0.5714,0.6429,0.7143,0.7857,0.9286,1.0000], dtype='float64')
        y_train = np.mat([6.8053,-1.5184,1.6416,6.3543,14.3442,16.4426,18.1953,28.9913,27.2246,40.3759,55.3726,72.0], dtype='float64')
        x_train = np.reshape(x_train, (M, 1))
        y_train = np.reshape(y_train, (M, 1))
        print len(x_train), len(y_train)
        myBasis = Basis('Univariate')
        poly = Polyreg(myParameters, myBasis, training_inputs=x_train, training_outputs=y_train)
        print poly.getfitStatistics()


if __name__== '__main__':
    unittest.main()
