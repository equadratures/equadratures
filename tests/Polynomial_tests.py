#!/usr/bin/env python
from effective_quadratures.parameter import Parameter
from effective_quadratures.indexset import IndexSet
from effective_quadratures.polynomial import Polynomial
from effective_quadratures.utils import meshgrid
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def main():

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

main()