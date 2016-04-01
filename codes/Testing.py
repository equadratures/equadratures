#!/usr/bin/python
import PolyMethod as poly
from PolyParams import PolynomialParam
import numpy as np
import matplotlib.pyplot as plt

"""
    Testing Class

"""

def main():

    uq_parameter1 = PolynomialParam("Jacobi", -1, 1.0, 0, 0) # Uniform parameter on [-,1,1]
    V = [uq_parameter1, uq_parameter1] # Two such params
    order = [100,100] # Has to be in brackets!
    K, I, F = poly.getPseudospectralCoefficients(V, order, function)

    # Sorting
    y, x = np.mgrid[0:order[0], 0:order[1] ]
    z = np.reshape(K,(order[0], order[1]))
    poly_order = np.sum(np.array(I),axis=1)

    # Plot of tensor grid pseudospectral coefficients
    plt.pcolor(y,x,np.log10(np.abs(z)), cmap='jet', vmin=-14, vmax=0)
    plt.title('Tensor grid pseudospectral coefficients')
    plt.xlabel('i1')
    plt.ylabel('i2')
    plt.colorbar()
    plt.xlim(0,order[0])
    plt.ylim(0,order[1])
    plt.show()

    # Plot of coefficient decay
    plt.scatter(poly_order, np.log10(np.abs(K)), s=20, c='r', marker='o')
    plt.xlabel('Polynomial order')
    plt.ylabel('Coefficient magnitude (log-scale)')
    plt.ylim(-16, 10)
    plt.xlim(0, order[0])
    plt.show()

def function(x):
    #return np.exp(15*x[0] + x[1])
    return 1.0/(2 + 16*(x[0] - 0.1)**2 + 25*(x[1] + 0.1)**2)
    #return np.sin(5*(x[0] - 0.5)) + np.cos(3*(x[1] - 1))
main()
