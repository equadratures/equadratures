#!/usr/bin/python
from PolyParams import PolynomialParam
import Integration as integrals
import matplotlib.pyplot as plt
import numpy as np
"""
    Example script for plotting sparse and tensor grids.
"""

def main():

    # Setup the parameters
    uq_parameter1 = PolynomialParam("Jacobi", -1, 1.0, 0, 0) # Uniform parameter on [-,1,1]
    V = [uq_parameter1, uq_parameter1] # Two such parameters

    # Inputs for a sparse grid:
    sparse_growth_rule = 'linear'
    sparse_level = 3
    pts, wts = integrals.SparseGrid(V, sparse_level, sparse_growth_rule)
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


main()
