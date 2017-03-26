#!/usr/bin/env python
from unittest import TestCase
import unittest
from equadratures import *
import numpy as np

class TestPolyreg(TestCase):

    def test_linear(self):

        x_train = np.mat([[0.0, 32.1205],
        [0.0714, 29.7601],
        [0.1429, 30.6798],
        [0.2023, 22.1313],
        [0.2857, 27.6573],
        [0.3571, 26.5302],
        [0.4001, 18.63],
        [0.4286, 25.0],
        [0.5714, 23.1148],
        [0.6429, 21.2932],
        [0.7143, 22.4207],
        [0.7857, 21.1507],
        [0.8310, 22.91923],
        [0.9286, 19.5951],
        [1.0000, 18.022]], dtype='float64')
        y_train = np.mat([6.8053,-1.5184,1.6416,2.0121,6.3543,14.3442,25.3121,16.4426,18.1953,28.9913,27.2246,40.3759,45.12322, 55.3726,72.0], dtype='float64')
        poly1 = Polyreg(x_train, y_train.T, 'quadratic')

if __name__ == '__main__':
    unittest.main()
