#!/usr/bin/env python
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
    method = "tensor grid"
    level = 2
    growth_rule = "linear"
    # Write out the properties for each "uq_parameter". You can have as many
    # as you like!
    uq_parameters = []
    uq_parameter1_to_3 = PolynomialParam("Uniform", min_value, max_value, parameter_A, parameter_B, derivative_flag, order)
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
    X , I, F = PolyParent.getCoefficients(uq_structure, fun)
    print '---Pseudospectral coefficients---'
    print X, I
    print '\n'
    print 'Mean: '+str(X[0,0])
    print 'Variance: '+str(np.sum(X[0,1:]**2))

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
    
    
def main():
    
    
    uq_parameter1 = PolynomialParam("Jacobi", -1, 1.0, 0, 0) # Uniform parameter on [-,1,1]
    V = [uq_parameter1, uq_parameter1] # Two such params
    sparse_growth_rule = 'exponential'
    sparse_level = 7


    sparse_coefficients = spam.getSPAM_LSQRCoefficients(V, function, sparse_growth_rule, sparse_level)
    x,y,z, max_order = twoDgrid(sparse_coefficients)
    
    z = np.log10(np.abs(z))
    Zm = ma.masked_where(np.isnan(z),z)
    plt.pcolor(y,x, Zm, cmap='jet', vmin=-14, vmax=0)
    plt.title('SPAM LSQR coefficients')
    plt.xlabel('i1')
    plt.ylabel('i2')
    plt.colorbar()
    plt.xlim(0,max_order)
    plt.ylim(0,max_order)
    plt.show()
    
    
def lineup(coefficients, index_set):
    orders_length = len(index_set[0])
    coefficient_array = np.zeros((len(coefficients), orders_length +1))
    for i in range(0, len(coefficients)):
        coefficient_array[i,0] = coefficients[i]
        for j in range(0, orders_length):
            coefficient_array[i,j+1] =  index_set[i,j]
 
    return coefficient_array
     
     
     
# Function just to help plotting!
def twoDgrid(spam_coefficients):
    
    max_order = int( np.max(spam_coefficients[:,1], axis=0) )
    
    # Now create a tensor grid with this max. order
    y, x = np.mgrid[0:max_order, 0:max_order]
    z = (x*0 + y*0) + float('NaN')
     
    # Now for each grid point, cycle through spam_coefficients and see if
    # that grid point is present, if so, add the coefficient value to z.
    for i in range(0, max_order):
        for j in range(0, max_order):
            x_entry = x[i,j]
            y_entry = y[i,j]
            for k in range(0, len(spam_coefficients)):
                if(x_entry == spam_coefficients[k,1] and y_entry == spam_coefficients[k,2]):
                    z[i,j] = spam_coefficients[k,0]
                    
    return x,y,z, max_order


main()
