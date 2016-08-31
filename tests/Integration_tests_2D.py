#!/usr/bin/env python
from effective_quadratures.PolyParams import PolynomialParam
from effective_quadratures.PolyParentFile import PolyParent
from effective_quadratures.IndexSets import IndexSet
import effective_quadratures.Integrals as integrals
from effective_quadratures.EffectiveQuadSubsampling import EffectiveSubsampling
import effective_quadratures.MatrixRoutines as mat
import effective_quadratures.Utils as utils
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
"""
    Testing integration rules.
"""

def main():

    # Parameter ranges x1, x2, x3, x4
    parameter_ranges = ([-1,1], [-1,1])

    # Tensor grid parameters
    orders = [5,5]

    # Sparse grid parameters
    level = 2
    growth_rule = "linear"

    # Hyperbolic Cross parameters -- for effective quadratures
    q = 0.4

    # Get the integral approximations and the corresponding number of points
    sparse_grid_approx, sparse_grid_points = integrals.sparseGrid(parameter_ranges, level, growth_rule, function)
    tensor_grid_approx, tensor_grid_points = integrals.tensorGrid(parameter_ranges, orders, function)
    effectivequad_approx, effective_quad_points = integrals.effectivelySubsampledGrid(parameter_ranges, orders, q, function)

    # Plots!

    # Show outputs!
    print '----INTEGRALS----'
    print 'Tensor grid approximation: '+'\t'+str(tensor_grid_approx)+'\t'+'with # of points:'+str(len(tensor_grid_points))
    print 'Sparse grid approximation: '+'\t'+str(sparse_grid_approx)+'\t'+'with # of points:'+str(len(sparse_grid_points))
    print 'Effectively subsampled grid approximation: '+'\t'+str(effectivequad_approx)+'\t'+'with # of points:'+str(len(effective_quad_points))
    label_size = 25
    mpl.rcParams['xtick.labelsize'] = label_size
    mpl.rcParams['ytick.labelsize'] = label_size
    plt.scatter(sparse_grid_points[:,0], sparse_grid_points[:,1], s=230, c='b', marker='o')
    plt.xlabel(r'$x_1$', fontsize=28)
    plt.ylabel(r'$x_2$', fontsize=28)
    plt.savefig('sparse.pdf', format='pdf', dpi=1000, bbox_inches='tight')
    plt.show()

    plt.scatter(tensor_grid_points[:,0], tensor_grid_points[:,1], s=230, c='r', marker='o')
    plt.xlabel(r'$x_1$', fontsize=28)
    plt.ylabel(r'$x_2$', fontsize=28)
    plt.savefig('tensor.pdf', format='pdf', dpi=1000, bbox_inches='tight')
    plt.show()

    plt.scatter(effective_quad_points[:,0], effective_quad_points[:,1], s=230, c='k', marker='o')
    plt.xlabel(r'$x_1$', fontsize=28)
    plt.ylabel(r'$x_2$', fontsize=28)
    plt.savefig('effective.pdf', format='pdf', dpi=1000, bbox_inches='tight')
    plt.show()

# Model or function!
def function(x):
    return x[0]**3 + 2*x[1]**2 + 2

main()
