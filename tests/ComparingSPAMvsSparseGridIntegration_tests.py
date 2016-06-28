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

    # Uq parameters setup.
    order = 5
    derivative_flag = 0 # derivative flag
    min_value = -1
    max_value = 1
    parameter_A = 0
    parameter_B = 0

    first_parameter = PolynomialParam("Uniform", min_value, max_value, parameter_A, parameter_B, derivative_flag, order)
    second_parameter = PolynomialParam("Uniform", min_value, max_value, parameter_A, parameter_B, derivative_flag, order)
    uq_parameters = [first_parameter, second_parameter]

    # Index set setup
    sparsegridObject = IndexSet("sparse grid", [], 6, "exponential", 2)


    # Create a PolyParent object!
    uq_sparse_integration = PolyParent(uq_parameters, "sparse grid", sparsegridObject)
    uq_spam = PolyParent(uq_parameters, "spam", sparsegridObject)

    # For coefficients!
    X , I  = PolyParent.getCoefficients(uq_sparse_integration, fun)
    X2, I2 = PolyParent.getCoefficients(uq_spam, fun)

    # Compute stats.
    mean, variance = stats.compute_mean_variance(X,I)
    print mean, variance
    mean, variance = stats.compute_mean_variance(X2,I2)
    print mean, variance
    """~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                    PLOTTING SECTION
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
    x,y,z, max_order = utils.twoDgrid(X,I)
    x2,y2,z2, max_order2 = utils.twoDgrid(X2,I2)

    z = np.log10(np.abs(z))
    z2 = np.log10(np.abs(z2))

    Zm = np.ma.masked_where(np.isnan(z),z)
    Zm2 = np.ma.masked_where(np.isnan(z2),z2)


    plt.subplot(121)
    plt.pcolor(x,y, Zm, cmap='jet', vmin=-15, vmax=0)
    plt.title('Sparse grid integration')
    plt.xlabel('i1')
    plt.ylabel('i2')
    plt.xlim(0,max_order)
    plt.ylim(0,max_order)
    plt.colorbar()
    plt.subplot(122)
    plt.pcolor(x2,y2, Zm2, cmap='jet', vmin=-15, vmax=0)
    plt.title('Sparse pseudospectral method')
    plt.xlabel('i1')
    plt.ylabel('i2')
    plt.colorbar()
    plt.xlim(0,max_order)
    plt.ylim(0,max_order)
    plt.savefig('compare.pdf', format='pdf')
    plt.show()



main()
