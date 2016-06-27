#!/usr/bin/env python
from effective_quadratures.PolyParams import PolynomialParam
from effective_quadratures.PolyParentFile import PolyParent
from effective_quadratures.IndexSets import IndexSet
import effective_quadratures.MatrixRoutines as matrix
import effective_quadratures.ComputeStats as stats
import effective_quadratures.Utils as utils
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
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
    order = 4
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
    level = 7
    growth_rule = "exponential"

    # Write out the properties for each "uq_parameter". You can have as many
    # as you like!
    uq_parameters = []
    uq_parameter1_to_3 = PolynomialParam("Uniform", min_value, max_value, parameter_A, parameter_B, derivative_flag, order)
    uq_parameters.append(uq_parameter1_to_3)
    uq_parameters.append(uq_parameter1_to_3)


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

    """
    pts, wts = PolyParent.getPointsAndWeights(uq_structure, level, growth_rule)

    print '--Quadrature Points--'
    print pts
    print '\n'
    print '--Weights--'
    print wts
    print '\n'
    """
    # For coefficients!
    X , I  = PolyParent.getCoefficients(uq_structure, fun)
    mean, variance = stats.compute_mean_variance(X,I)
    print '---Pseudospectral coefficients---'
    #print X
    print '---Index set---'
    #print I
    print '\n'
    print 'Mean: '+str(mean)
    print 'Variance: '+str(variance)

    """~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                    PLOTTING SECTION
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
    x,y,z, max_order = utils.twoDgrid(X,I)

    z = np.log10(np.abs(z))
    Zm = np.ma.masked_where(np.isnan(z),z)
    plt.pcolor(y,x, Zm, cmap='jet', vmin=-14, vmax=0)
    plt.title('Pseudospectral coefficients')
    plt.xlabel('i1')
    plt.ylabel('i2')
    plt.colorbar()
    plt.xlim(0,max_order)
    plt.ylim(0,max_order)
    plt.show()


main()
