#!/usr/bin/env python
from effective_quadratures.IndexSets import IndexSet
import effective_quadratures.MatrixRoutines as matrix
from effective_quadratures.EffectiveQuadSubsampling import EffectiveSubsampling
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os
"""

    Testing Script for Effective Quadrature Suite of Tools

    Pranay Seshadri
    ps583@cam.ac.uk

    Copyright (c) 2016 by Pranay Seshadri
"""

def plot_index_sets_2D():
    order = [4,4]
    q_parameter = 0.5

    # Hyperbolic cross basis
    hyperbolic_basis = IndexSet("hyperbolic cross", order, q_parameter)
    index_set = IndexSet.getIndexSet(hyperbolic_basis)

    # Tensor grid basis
    tensor_grid_basis = IndexSet("tensor grid",  order)
    tensor_set = IndexSet.getIndexSet(tensor_grid_basis)

    # Plot index set!
    x_plot = column(index_set, 0)
    y_plot = column(index_set, 1)
    ten_x = column(tensor_set, 0)
    ten_y = column(tensor_set, 1)


    plt.scatter(ten_x, ten_y, marker='x', s=90, color='red')
    plt.scatter(x_plot, y_plot, s=80)
    plt.xlabel('x index')
    plt.ylabel('y index')
    plt.xlim((-0.05,4.05))
    plt.ylim((-0.05,4.05))
    plt.show()

def plot_index_sets_3D():
    order = [4,4,4]
    q_parameter = 0.7

    # Hyperbolic cross basis
    hyperbolic_basis = IndexSet("hyperbolic cross", order, q_parameter)
    index_set = IndexSet.getIndexSet(hyperbolic_basis)

    # Tensor grid basis
    tensor_grid_basis = IndexSet("tensor grid",  order)
    tensor_set = IndexSet.getIndexSet(tensor_grid_basis)

    # Plot index set!
    x_plot = column(index_set, 0)
    y_plot = column(index_set, 1)
    z_plot = column(index_set, 2)
    ten_x = column(tensor_set, 0)
    ten_y = column(tensor_set, 1)
    ten_z = column(tensor_set, 2)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(ten_x, ten_y, ten_z, marker='x', s=90, color='red')
    ax.scatter(x_plot, y_plot, z_plot, s=80)
    ax.set_xlabel('i1')
    ax.set_ylabel('i2')
    ax.set_zlabel('i3')
    plt.xlim((-0.05,4.05))
    plt.ylim((-0.05,4.05))
    plt.title('Index set in 3D')
    plt.show()


def column(matrix, i):
    return [row[i] for row in matrix]

plot_index_sets_3D()
