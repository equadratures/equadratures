#!/usr/bin/env python
from effective_quadratures.PolyParams import PolynomialParam
from effective_quadratures.PolyParentFile import PolyParent
from effective_quadratures.IndexSets import IndexSet
import effective_quadratures.Integrals as integrals
import effective_quadratures.Utils as utils
import matplotlib.pyplot as plt
import numpy as np
"""
    Testing integration rules.
"""

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

    # Index set setup - don't need one for a tensor grid...but do need one for a sparse grid.
    tensorgridObject = IndexSet("tensor grid", [order, order])
    sparsegridObject = IndexSet("sparse grid", [], 7, "exponential", 2)

    # Get the points and weights!
    sg_pts, sg_wts = integrals.sparseGrid(uq_parameters, sparsegridObject)
    tg_pts, tg_wts = integrals.tensorGrid(uq_parameters, tensorgridObject)

    tensor_int = np.mat(tg_wts) * utils.evalfunction(tg_pts, function)
    sparse_int = np.mat(sg_wts) * utils.evalfunction(sg_pts, function)

    print 'Integrals'
    print tensor_int
    print sparse_int

    # Plot sparse grid
    plt.scatter(sg_pts[:,0], sg_pts[:,1], s=70, c='r', marker='o')
    plt.xlabel('first parameter')
    plt.ylabel('second parameter')
    plt.title('Sparse grid points')
    plt.xlim(min_value,max_value)
    plt.ylim(min_value,max_value)
    plt.show()

    # Plot tensor grid
    plt.scatter(tg_pts[:,0], tg_pts[:,1], s=70, c='b', marker='o')
    plt.xlabel('first parameter')
    plt.ylabel('second_parameter')
    plt.title('Tensor points')
    plt.xlim(min_value,max_value)
    plt.ylim(min_value,max_value)
    plt.show()


# Model or function!
def function(x):
    return np.exp(x[0] + x[1])

main()
