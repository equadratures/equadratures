#!/usr/bin/env python
from effective_quadratures.parameter import Parameter
from effective_quadratures.polynomial import Polynomial
from effective_quadratures.indexset import IndexSet
from effective_quadratures.effectivequads import EffectiveSubsampling
from effective_quadratures.computestats import Statistics
import numpy as np

def main():

    # Vegetation study example Test
    x1 = Parameter(lower=38.2, upper=250.4, points=3)
    x2 = Parameter(lower=0.157, upper=0.313, points=3)
    x3 = Parameter(lower=0.002, upper=0.01,  points=3)
    x4 = Parameter(lower=0.0002, upper=0.001, points=3)
    parameters = [x1, x2, x3, x4]

    orders = [2, 2, 2, 2]
    polybasis = IndexSet("Total order", orders)
    print polybasis.getIndexSet()
    maximum_number_of_evals = polybasis.getCardinality()

    esq = EffectiveSubsampling(parameters, polybasis)
    points = esq.getPointsToEvaluate(maximum_number_of_evals)
    print points

    Output = [15.9881,16.5091,16.0162,15.9950,16.0310,16.4592,16.0958,15.8507,16.0757,15.9252,16.4301,16.1259,16.4682,16.0501,16.2200]
    Output = np.mat(Output)

    x = esq.solveLeastSquares(maximum_number_of_evals, Output.T)
    print x

    vegeUQ = Statistics(x, polybasis)
    mean = vegeUQ.getMean()
    variance = vegeUQ.getVariance()
    sobol = vegeUQ.getFirstOrderSobol()

    print mean, variance
    print sobol
main()