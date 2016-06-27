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
    order = 8
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
    uq_structure = PolyParent(uq_parameters, "sparse grid", sparsegridObject)

    # For coefficients!
    X , I  = PolyParent.getCoefficients(uq_structure, fun)

    # Compute stats.
    mean, variance = stats.compute_mean_variance(X,I)

    """~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                    PLOTTING SECTION
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
    x,y,z, max_order = utils.twoDgrid(X,I)

    z = np.log10(np.abs(z))
    Zm = np.ma.masked_where(np.isnan(z),z)
    plt.pcolor(y,x, Zm, cmap='jet', vmin=-13, vmax=0)
    plt.title('Pseudospectral coefficients')
    plt.xlabel('i1')
    plt.ylabel('i2')
    plt.colorbar()
    plt.xlim(0,max_order)
    plt.ylim(0,max_order)
    plt.show()


main()
