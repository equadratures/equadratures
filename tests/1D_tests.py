#!/usr/bin/env python
from effective_quadratures.PolyParams import PolynomialParam
from effective_quadratures.PolyParentFile import PolyParent
from effective_quadratures.IndexSets import IndexSet
import effective_quadratures.ComputeStats as stats
import effective_quadratures.MatrixRoutines as matrix
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import numpy.ma as maogle
import os
"""

    Script for Vegetation Sensitivity Study

    Pranay Seshadri
    ps583@cam.ac.uk

    Copyright (c) 2016 by Pranay Seshadri
"""
# Simple analytical function
def fun(x):
    #return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
    return x[0]**2 - 13*x[0] + 15
def main():

    """~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                    INPUT SECTION
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
    order = 20
    derivative_flag = 0 # derivative flag
    error_flag = 0

    # Min and max values. Not used for a "Gaussian" or "Normal" distribution
    min_value = 1
    max_value = 4

    # For a "Beta" uncertainty, these become alpha and beta shape parameters
    # in which case both have to be greater than 1.0
    # For a "Normal" or "Gaussian" uncertainty these become the mean and variance
    parameter_A = 2
    parameter_B = 3

    # Method for computing coefficients. Right now functionality is limited to
    # tensor grids. to do: THIS NEEDS TO BE CODED
    method = "tensor grid"

    # Write out the properties for each "uq_parameter". You can have as many
    # as you like!
    uq_parameters = []
    uq_parameter = PolynomialParam("Gaussian", min_value, max_value, parameter_A, parameter_B, derivative_flag, order)
    uq_parameters.append(uq_parameter)
    #uq_parameters.append(uq_parameter)


    print '****************************************************************'
    print '                     EFFECTIVE-QUADRATURES                      '
    print '\n'
    for i in range(0,len(uq_parameters)):
        print str('Uncertainty Parameter %i : '%(i+1)) + str(uq_parameters[i].param_type)
        if(uq_parameters[i].param_type == "Gaussian" or uq_parameters[i].param_type == "Normal"):
            print str('With mean & variance:')+'\t'+('[')+str(uq_parameters[i].shape_parameter_A)+str(',')+str(uq_parameters[i].shape_parameter_B)+str(']')
        elif(uq_parameters[i].param_type == "Beta" ):
            print str('With shape parameters:')+'\t'+('[')+str(uq_parameters[i].shape_parameter_A)+str(',')+str(uq_parameters[i].shape_parameter_B)+str(']')
        elif(uq_parameters[i].param_type == "Beta" or uq_parameters[i].param_type == "Uniform"):
            print str('With support:')+'\t'+('[')+str(uq_parameters[i].lower_bound)+str(',')+str(uq_parameters[i].upper_bound)+str(']')
            print str('Order:')+'\t'+str(uq_parameters[i].order)+'\n'
    print '****************************************************************'

    # Create a PolyParent object!
    uq_structure = PolyParent(uq_parameters, method)
    pts, wts = PolyParent.getPointsAndWeights(uq_structure)

    print '--Quadrature Points--'
    print pts
    print '\n'
    print '--Weights--'
    print wts
    print '\n'

    # For coefficients!
    X , I, F = PolyParent.getCoefficients(uq_structure, fun)
    print X, I

    # Get Sobol indices!
    mean, variance = stats.compute_mean_variance(X, I)
    #sobol = stats.compute_first_order_Sobol_indices(X, I)
    print mean, variance
    #print sobol
    #print 'Finished!'



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
