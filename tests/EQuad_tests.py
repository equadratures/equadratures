#!/usr/bin/env python
from effective_quadratures.parameter import Parameter
from effective_quadratures.polynomial import Polynomial
from effective_quadratures.indexset import IndexSet
from effective_quadratures.effectivequads import EffectiveSubsampling
from effective_quadratures.computestats import Statistics
import numpy as np

# Vegetation study example Test
def test():

    # Set the parameters
    x1 = Parameter(lower=38.2, upper=250.4, points=3)
    x2 = Parameter(lower=0.157, upper=0.313, points=3)
    x3 = Parameter(lower=0.002, upper=0.01,  points=3)
    x4 = Parameter(lower=0.0002, upper=0.001, points=3)
    parameters = [x1, x2, x3, x4]

    # Set the polynomial basis
    orders = [2, 2, 2, 2]
    polybasis = IndexSet("Total order", orders)
    print polybasis.getIndexSet()
    maximum_number_of_evals = polybasis.getCardinality()

    # Set up effective quadrature subsampling
    esq = EffectiveSubsampling(parameters, polybasis)
    points = esq.getEffectivelySubsampledPoints(maximum_number_of_evals)
    print points

    # Use the output from simulation data
    #Output = [15.9881,16.5091,16.0162,15.9950,16.0310,16.4592,16.0958,15.8507,16.0757,15.9252,16.4301,16.1259,16.4682,16.0501,16.2200]
    Output = [0.0906050857157,0.0776969827712,0.0864368518814,0.0932615157217,0.0892242211848,0.0767011023127,0.0866207387298, 0.0977708660066,0.0861118221655,0.0963280722499,0.0774124991149,0.087565776892,0.0768618592992,0.0870198933408,0.0866443598643]
    #Output = [24.8170119614, 23.7770471604, 24.6131673073, 24.9723698096, 24.6920894782, 23.7015914415, 24.6180488646, 25.2502586048, 24.5767649971, 25.1656386185, 23.7951837649, 24.5402057883, 23.7204920408, 24.5140496633, 24.4823906956]
    #Output = [9.28511113565, 8.59116120376, 9.20725286751, 9.22159875619, 9.19565064461, 8.51969919061, 9.15223514391, 9.38391902559, 9.15666537065	, 9.31338865761	, 8.57543965821	, 9.0772799181, 8.53049860903, 9.17308676441, 8.98211942423]
    Output = np.mat(Output)

    # Solve the least squares problem
    x = esq.solveLeastSquares(maximum_number_of_evals, Output.T)
    print x

    # Compute statistics!
    vegeUQ = Statistics(x, polybasis)
    mean = vegeUQ.getMean()
    variance = vegeUQ.getVariance()
    sobol = vegeUQ.getFirstOrderSobol()
    print mean, variance
    print sobol


test()
