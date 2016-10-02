#!/usr/bin/env python
from effective_quadratures.parameter import Parameter
from effective_quadratures.indexset import IndexSet
from effective_quadratures.polynomial import Polynomial
from effective_quadratures.utils import meshgrid
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def main():

    # Plot the first few Legendre orthonormal polynomials and their derivatives!
    # 1D case!************************************************************
    s = Parameter(lower=-1, upper=2, param_type='Uniform', points=4, derivative_flag=1)
    T = IndexSet('Tensor grid', [5])
    uq = Polynomial([s], method='Tensor', index_sets=T)
    pts = np.linspace(-1, 1, 20)
    P , D = uq.getMultivariatePolynomial(pts)
    print P
    print '\n'
    print D

    # 2D Case!************************************************************
    s = Parameter(lower=-2, upper=2, param_type='Uniform', points=3)
    T = IndexSet('Tensor grid', [3,3])
    uq_parameters = [s,s]
    uq = Polynomial(uq_parameters, method='Tensor', index_sets=T)

    # Creats an array of pts in 2D
    num_elements = 10
    pts, x1, x2 = meshgrid(-1.0, 1.0, num_elements,num_elements)

    # Now get the Polynomial
    P = uq.getMultivariatePolynomial(pts)
    Z = np.reshape(P, (num_elements,num_elements) )

    # Plot!
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #Axes3D.plot_surface(ax, x1, x2, Z,cmap=plt.cm.jet, rstride=1, cstride=1, linewidth=0, alpha=0.5)
    #ax.set_xlabel('x1')
    #ax.set_ylabel('x2')
    #plt.show()

main()