#!/usr/bin/env python
from effective_quadratures.parameter import Parameter
from effective_quadratures.polynomial import Polynomial
from effective_quadratures.indexset import IndexSet
from effective_quadratures.effectivequads import EffectiveSubsampling
from effective_quadratures.computestats import Statistics
import numpy as np
# Vegetation study example Test
def main():

    # Set the parameters
    x1 = Parameter(lower=38.2, upper=250.4, points=3)
    x2 = Parameter(lower=0.157, upper=0.313, points=3)
    parameters = [x1, x2]

    # Set the polynomial basis
    orders = [2, 2]
    polybasis = IndexSet("Total order", orders)
    print polybasis.getIndexSet()
    maximum_number_of_evals = polybasis.getCardinality()

    # Set up effective quadrature subsampling
    esq = EffectiveSubsampling(parameters, polybasis)
    Asquare = esq.getAsubsampled(maximum_number_of_evals)
    print Asquare

    # Solve the least squares problem and compare the result to 
    # what we know!
    

main()
