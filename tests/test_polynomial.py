#!/usr/bin/env python
from unittest import TestCase
import unittest
from effective_quadratures.parameter import Parameter
from effective_quadratures.indexset import IndexSet
from effective_quadratures.polynomial import Polynomial
from effective_quadratures.utils import meshgrid, twoDgrid, evalfunction
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy.ma as ma

class TestParameter(TestCase):
    
    def test_polynomial_and_derivative_constructions(self):
        s = Parameter(lower=-1, upper=1, param_type='Uniform', points=2, derivative_flag=1)
        uq_parameters = [s,s]
        uq = Polynomial(uq_parameters)
        num_elements = 2
        pts, x1, x2 = meshgrid(-1.0, 1.0, num_elements,num_elements)
     
        P , Q = uq.getMultivariatePolynomial(pts)
        print '--------output---------'
        print P
        print '~'
        print Q

        s = Parameter(lower=-1, upper=2, param_type='Uniform', points=4, derivative_flag=0)
        T = IndexSet('Tensor grid', [5])
        uq = Polynomial([s])
        pts = np.linspace(-1, 1, 20)
        P , D = uq.getMultivariatePolynomial(pts)
        print '--------output---------'
        print P 
        print '~'
        print D

    def test_pseudospectral_coefficient_routines(self):
        def expfun(x):
            return np.exp(x[0] +  x[1])

        s = Parameter(lower=-1, upper=1, points=5)
        T = IndexSet('Sparse grid', level=3, growth_rule='linear', dimension=2)
        uq = Polynomial([s,s], T)
        coefficients, index_set, evaled_pts = uq.getPolynomialCoefficients(expfun)
        x,y,z, max_order = twoDgrid(coefficients, index_set)
        z = np.log10(np.abs(z))

        # Plot of the pseudospectral coefficients
        Zm = ma.masked_where(np.isnan(z),z)
        plt.pcolor(y,x, Zm, cmap='jet', vmin=-14, vmax=0)
        plt.title('SPAM coefficients')
        plt.xlabel('i1')
        plt.ylabel('i2')
        plt.colorbar()
        plt.xlim(0,max_order)
        plt.ylim(0,max_order)

        # Plot of the sparse grid points
        plt.plot(evaled_pts[:,0], evaled_pts[:,1], 'ro')
    
    def test_pseudospectral_approximation_tensor(self):
        
        def expfun(x):
            return np.exp(x[0] + x[1] ) + 0.5*np.cos(x[0]*2*np.pi)

        # Compare actual function with polynomial approximation
        s = Parameter(lower=-1, upper=1, points=6)
        T = IndexSet('Tensor grid', [5,5])
        uq = Polynomial([s,s], T)
        num_elements = 10
        coefficients, index_set, evaled_pts = uq.getPolynomialCoefficients(expfun)
        pts, x1, x2 = meshgrid(-1.0, 1.0, num_elements,num_elements)
        Approx = uq.getPolynomialApproximation(expfun, pts, coefficients)
        A = np.reshape(Approx, (num_elements,num_elements))
        fun = evalfunction(pts, expfun)

        # Now plot this surface
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(x1, x2, A, rstride=1, cstride=1, cmap=cm.winter,
                       linewidth=0, antialiased=False,  alpha=0.5)
        ax.scatter(x1, x2, fun, 'ko')
        ax.set_zlim(0, 10)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('Response')

        fig.colorbar(surf, shrink=0.5, aspect=5)
        #plt.show()
    
    def test_pseudospectral_approximation_spam(self):
            
        def expfun(x):
            return np.exp(x[0] + x[1] ) + 0.5*np.cos(x[0]*2*np.pi)

        # Compare actual function with polynomial approximation
        s = Parameter(lower=-1, upper=1, points=6)
        T = IndexSet('Sparse grid', level=3, growth_rule='linear', dimension=2)
        uq = Polynomial([s,s], T)
        num_elements = 10
        coefficients, index_set, evaled_pts = uq.getPolynomialCoefficients(expfun)
        pts, x1, x2 = meshgrid(-1.0, 1.0, num_elements,num_elements)
        Approx = uq.getPolynomialApproximation(expfun, pts)
        A = np.reshape(Approx, (num_elements,num_elements))
        fun = evalfunction(evaled_pts, expfun)

        # Now plot this surface
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(x1, x2, A, rstride=1, cstride=1, cmap=cm.winter,
                       linewidth=0, antialiased=False,  alpha=0.5)
        ax.scatter(evaled_pts[:,0], evaled_pts[:,1], fun, 'ko')
        ax.set_zlim(0, 10)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('Response')

        fig.colorbar(surf, shrink=0.5, aspect=5)
        #plt.show()


if __name__ == '__main__':
    unittest.main()