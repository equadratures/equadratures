from equadratures import *
import numpy as np


def very_complex_model(x):
    return (np.abs(x[0] - 0.2) + np.abs(x[1] + 0.2) )**3

def main():

    sparse = IndexSet('Sparse grid', level=7, growth_rule='exponential',dimension=2)
    x1 = Parameter(param_type="Uniform", lower=-1, upper=1, points=10)
    spam = Polyint([x1,x1], sparse)
    coefficients, index_set, pts = spam.getPolynomialCoefficients(very_complex_model)
    #plotting.coeffplot2D(coefficients, index_set, '$i_1$', '$i_2$')

main()