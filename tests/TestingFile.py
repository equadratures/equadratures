#!/usr/bin/python
from effective_quadratures.PolyParams import PolynomialParam
from effective_quadratures.PolyParentFile import PolyParent
from effective_quadratures.IndexSets import IndexSet
import effective_quadratures.MatrixRoutines as matrix
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import numpy.ma as maogle
import os
"""

    Testing Script for Effective Quadrature Suite of Tools

    Pranay Seshadri
    ps583@cam.ac.uk

    Copyright (c) 2016 by Pranay Seshadri
"""
# Simple analytical function
def fun(x):
    return x[:]

def main():

    """~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                    INPUT SECTION

    NOTES:
        1. min_value and max_value are not used when param_type = "Normal" or
        "Gaussian".
        2. parameter_A and parameter_B are the shape parameters when param_type
        is "Jacobi"; alpha and beta respectively. For a normal distribution
        these become the mean and the standard deviation.
        3. The normal distribution is for a mean of 0 and variance of 0.5 by
        default. For all others adjust using variance = param_A + 0.5
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
    order = 4
    derivative_flag = 0 # derivative flag on=1; off=0
    error_flag = 0
    min_value, max_value = 0, 1

    # What it should be
    parameter_A = 0
    parameter_B = 0

    uq_parameter1 = PolynomialParam("Gaussian", min_value, max_value, parameter_A, parameter_B, derivative_flag, order) # Setup uq_parameter
    uq_parameters = [uq_parameter1]
    pts_for_plotting = np.linspace(min_value, max_value, 600)
    indexset_configure = IndexSet("total order", [order])
    print '****************************************************************'
    print 'MIN'+'\t'+'MAX'+'\t'+'ALPHA'+'\t'+'BETA'+'\t'+'PTS'+'\t'+'FUNCTION'+'\n'
    print str(min_value)+'\t'+str(max_value)+'\t'+str(parameter_A)+'\t'+str(parameter_B)+'\t'+str(order)+'\t'+str('f(x) = x')
    print '****************************************************************'

    # Compute elements of an index set:self, index_set_type, orders, level=None, growth_rule=None):
    indices = IndexSet.getIndexSet(indexset_configure)

    # Create a PolyParent object!
    uq_structure = PolyParent(uq_parameters, indices)
    pts, wts = PolyParent.getTensorQuadrature(uq_structure)

    """~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                 PRINT OUTPUTS TO SCREEN
                           *In the future write output to file!
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
    print '--Quadrature Points--'
    print pts
    print '\n'
    print '--Weights--'
    print wts
    print '\n'

    # For coefficients!
    X , T = PolyParent.getCoefficients(uq_structure, fun)
    print '---Pseudospectral coefficients---'
    print X
    print '\n'
    print 'Mean: '+str(X[0,0])
    print 'Variance: '+str(np.sum(X[0,1:]**2))

    # Computing int(poly^2) * w
    #P = PolyParent.getMultivariatePoly(uq_structure, pts)
    #poly_value_1 = np.sum( P[0,:]**2 * wts)
    #poly_value_2 = np.sum( P[1,:]**2 * wts)
    #poly_value_3 = np.sum( P[2,:]**2 * wts)
    #print poly_value_1, poly_value_2, poly_value_3

    """~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                    PLOTTING SECTION
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
    # Plot all the univariate polynomials:
    #M = PolyParent.getMultivariatePoly(uq_structure, pts_for_plotting)
    #color=iter(cm.rainbow(np.linspace(0,1,order)))

    #for i in range(0, order):
    #    c = next(color)
    #    plt.plot(pts_for_plotting, M[i,:], '-', c=c)
    #plt.xlabel('x')
    #plt.ylabel('p(x)')
    #plt.title('Orthogonal polynomials')
    #plt.show()

main()
