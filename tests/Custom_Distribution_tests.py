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

    Test script for a custom distribution

    Pranay Seshadri
    ps583@cam.ac.uk

    Copyright (c) 2016 by Pranay Seshadri
"""

# Simple analytical function
def fun(x):
    #return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
    return x[0]

def main():

    """~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                    INPUT SECTION
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
    order = 6
    derivative_flag = 0 # derivative flag
    error_flag = 0

    # Min and max values. Not used for a "Gaussian" or "Normal" distribution
    min_value = 1
    max_value = 4

    # For a "Beta" uncertainty, these become alpha and beta shape parameters
    # in which case both have to be greater than 1.0
    # For a "Normal" or "Gaussian" uncertainty these become the mean and variance
    mean = 13
    variance = 12

    # Method for computing coefficients. Right now functionality is limited to
    # tensor grids. to do: THIS NEEDS TO BE CODED
    method = "tensor grid"

    # Write out the properties for each "uq_parameter". You can have as many
    # as you like!
    uq_parameters = []
    uq_parameter = PolynomialParam("FunGaussian", min_value, max_value, mean, variance, derivative_flag, order)
    uq_parameters.append(uq_parameter)


    # Create a PolyParent object!
    uq_structure = PolyParent(uq_parameters, method)
    pts, wts = PolyParent.getPointsAndWeights(uq_structure)
    print '--- Points and Weights ---'
    print pts, wts
    # For coefficients!
    X , I, F = PolyParent.getCoefficients(uq_structure, fun)
    print X, I

    # Get Sobol indices!
    mean, variance = stats.compute_mean_variance(X, I)
    print mean, variance



main()
