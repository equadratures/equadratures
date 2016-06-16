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
    min_value, max_value = -1, 1
    parameter_A , parameter_B = 2 , 2



    # Write out the properties for each "uq_parameter". You can have as many
    # as you like!
    uq_parameters = []
    uq_parameter1_to_3 = PolynomialParam("Uniform", min_value, max_value, parameter_A, parameter_B, derivative_flag, order)
    uq_parameter4_to_6 = PolynomialParam("Beta", min_value, max_value, parameter_A, parameter_B, derivative_flag, order)
    uq_parameters.append(uq_parameter1_to_3)
    uq_parameters.append(uq_parameter1_to_3)


    # Create a PolyParent object!
    uq_structure = PolyParent(uq_parameters, method, level, growth_rule)
    pts, wts = PolyParent.getPointsAndWeights(uq_structure, level, growth_rule)





main()
