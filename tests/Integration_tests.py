#!/usr/bin/env python
from effective_quadratures.PolyParams import PolynomialParam
from effective_quadratures.PolyParentFile import PolyParent
from effective_quadratures.IndexSets import IndexSet
import effective_quadratures.Integrals as integrals
import effective_quadratures.Utils as utils
import numpy as np
"""
    Testing integration rules.
"""

def main():

    # Uq parameters setup.
    order = 4
    derivative_flag = 0 # derivative flag
    min_value = -1
    max_value = 1
    parameter_A = 0
    parameter_B = 0
    uq_parameters = []
    uq_parameterx = PolynomialParam("Uniform", min_value, max_value, parameter_A, parameter_B, derivative_flag, order)
    uq_parameters.append(uq_parameterx)
    uq_parameters.append(uq_parameterx)

    # Index set setup - don't need one for a tensor grid...but do need one for a sparse grid.
    tensorgridObject = PolyParent(uq_parameters, "tensor grid")
    sparsegridObject = PolyParent(uq_parameters, "sparse grid",  3, "exponential")

    # Get the points and weights!
    sg_pts, sg_wts = integrals.sparseGrid(sparsegridObject)
    tg_pts, tg_wts = integrals.tensorGrid(tensorgridObject)

    tensor_int = utils.evalfunction(tg_pts, function) * np.mat(tg_wts)
    sparse_int = utils.evalfunction(sg_pts, function) * np.mat(sg_wts)
    """
    wts = np.mat(wts)
    wts = wts.T

    # Inputs for a tensor grid
    tensor_orders = [7,7]
    pts2, wts2 = integrals.TensorGrid(V, tensor_orders)
    wts2 = np.mat(wts2)
    wts2 = wts2.T

    # Plot sparse grid
    plt.scatter(pts[:,0], pts[:,1], s=20, c='r', marker='o')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Sparse grid points')
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.show()

    # Plot tensor grid
    plt.scatter(pts2[:,0], pts2[:,1], s=20, c='b', marker='o')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Tensor points')
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.savefig('foo.pdf')
    plt.show()

    # Comparing integration:
    tensor_int = evalfunction(pts2) * wts2
    sparse_int = evalfunction(pts) * wts

    print(sparse_int)
    print(tensor_int)
    """
# Model or function!
def function(x):
    return np.exp(x[0] + x[1])

main()
