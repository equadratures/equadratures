#!/usr/bin/env python
from unittest import TestCase
import unittest
from effective_quadratures.polynomial import PolyFit
from effective_quadratures.utils import meshgrid
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

class TestPolyFit(TestCase):
    
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
        poly1 = PolyFit(x_train, y_train.T, 'quadratic')

        X1 = np.arange(-0.2, 1.2, 0.05)
        X2 = np.arange(16, 34, 0.5)
        xx1, xx2 = np.meshgrid(X1, X2)
        u, v = xx1.shape
        x_test = np.mat( np.hstack( [np.reshape(xx1, (u*v, 1)),  np.reshape(xx2, (u*v, 1)) ]) , dtype='float64')
        y_test = poly1.testPolynomial(x_test)
        yy1 = np.reshape(y_test, (u, v))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=31., azim=168)
        ax.plot_surface(xx1,xx2, yy1,rstride=1, cstride=1, cmap=cm.winter, linewidth=0.02, alpha=0.5)
        ax.scatter(x_train[:,0].A1, x_train[:,1].A1, y_train.A1, c='red', marker='o', s=120, alpha=0.3)
        plt.show()


if __name__ == '__main__':
    unittest.main()