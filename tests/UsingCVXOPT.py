#!/usr/bin/env python
from effective_quadratures.PolyParams import PolynomialParam
from effective_quadratures.PolyParentFile import PolyParent
from effective_quadratures.IndexSets import IndexSet
import effective_quadratures.MatrixRoutines as matrix
import effective_quadratures.ComputeStats as stats
from effective_quadratures.EffectiveQuadSubsampling import EffectiveSubsampling
import effective_quadratures.Utils as utils
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import numpy.ma as maogle
import os
"""

    Incorporating utilities from CVXOPT for figuring out whether the D-optimal
    points are similar to what QR with column pivoting yields

    Pranay Seshadri
    ps583@cam.ac.uk

    Copyright (c) 2016 by Pranay Seshadri
"""
def main():

    # Setup.
    order = 5
    derivative_flag = 0
    min_value = -1.0
    max_value = 1.0
    q_parameter = 1.00

    # Create hyperbolic cross space
    hyperbolic_basis = IndexSet("total order", [order-1, order-1])
    maximum_number_of_evals = IndexSet.getCardinality(hyperbolic_basis)
    index_elements = IndexSet.getIndexSet(hyperbolic_basis)

    # Parameter types and ranges
    parameter = PolynomialParam("Uniform", min_value, max_value, [], [] , derivative_flag, order)
    parameters = [parameter, parameter]

    # Define the EffectiveSubsampling object and get "A"
    effectiveQuads = EffectiveSubsampling(uq_parameters, hyperbolic_basis, derivative_flag)
    Afull = EffectiveSubsampling.ge
    A, pts, W, not_used = EffectiveSubsampling.getAsubsampled(effectiveQuads, maximum_number_of_evals)

    # Now we use CVXOPT


main()
