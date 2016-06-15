#!/usr/bin/python
from PolyParams import PolynomialParam
from PolyParentFile import PolyParent
from IndexSets import IndexSet
import MatrixRoutines as matrix
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
    return np.exp(x[0] + x[1])

def main():

    """~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                    INPUT SECTION
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
    order = 5
    derivative_flag = 0 # derivative flag
    error_flag = 0

    # Min and max values. Not used for a "Gaussian" or "Normal" distribution
    min_value = -1
    max_value = 1

    # For a "Beta" uncertainty, these become alpha and beta shape parameters
    # in which case both have to be greater than 1.0
    # For a "Normal" or "Gaussian" uncertainty these become the mean and variance
    parameter_A = 2
    parameter_B = 2

    # Method for computing coefficients. Right now functionality is limited to
    # tensor grids. to do: THIS NEEDS TO BE CODED
    method = "sparse grid"
    level = 2
    growth_rule = "exponential"
    # Write out the properties for each "uq_parameter". You can have as many
    # as you like!
    uq_parameters = []
    uq_parameter1_to_3 = PolynomialParam("Uniform", min_value, max_value, parameter_A, parameter_B, derivative_flag, order)
    uq_parameter4_to_6 = PolynomialParam("Beta", min_value, max_value, parameter_A, parameter_B, derivative_flag, order)
    uq_parameters.append(uq_parameter1_to_3)
    uq_parameters.append(uq_parameter1_to_3)
    #uq_parameters.append(uq_parameter1_to_3)
    #uq_parameters.append(uq_parameter4_to_6)
    #uq_parameters.append(uq_parameter4_to_6)
    #uq_parameters.append(uq_parameter4_to_6)
    #pts_for_plotting = np.linspace(min_value, max_value, 600)


    print '****************************************************************'
    print '                     EFFECTIVE-QUADRATURES                      '
    print '\n'
    for i in range(0,len(uq_parameters)):
        print str('Uncertainty Parameter %i : '%(i+1)) + str(uq_parameters[i].param_type)
        if(uq_parameters[i].param_type == "Gaussian" or uq_parameters[i].param_type == "Normal"):
            print str('With mean & variance:')+'\t'+('[')+str(uq_parameters[i].shape_parameter_A)+str(',')+str(uq_parameters[i].shape_parameter_B)+str(']')
        elif(uq_parameters[i].param_type == "Beta" ):
            print str('With shape parameters:')+'\t'+('[')+str(uq_parameters[i].shape_parameter_A)+str(',')+str(uq_parameters[i].shape_parameter_A)+str(']')
        elif(uq_parameters[i].param_type == "Beta" or uq_parameters[i].param_type == "Uniform"):
            print str('With support:')+'\t'+('[')+str(uq_parameters[i].lower_bound)+str(',')+str(uq_parameters[i].upper_bound)+str(']')
            print str('Order:')+'\t'+str(uq_parameters[i].order)+'\n'
    print '****************************************************************'

    # Create a PolyParent object!
    uq_structure = PolyParent(uq_parameters, method, level, growth_rule)
    #uq_structure = PolyParent(uq_parameters, "tensor grid")


    pts, wts = PolyParent.getPointsAndWeights(uq_structure, level, growth_rule)

    print '--Quadrature Points--'
    print pts
    print '\n'
    print '--Weights--'
    print wts
    print '\n'

    # For coefficients!
    X , F, T = PolyParent.getCoefficients(uq_structure, fun)
    print '---Pseudospectral coefficients---'
    print X
    print '\n'
    print 'Mean: '+str(X[0,0])
    print 'Variance: '+str(np.sum(X[0,1:]**2))
    print T

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
