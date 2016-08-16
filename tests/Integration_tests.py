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
"""
    Testing integration rules.
"""

def main():

    # Parameter ranges x1, x2, x3, x4
    parameter_ranges = ([-0.5, 0.5], [-1, 2], [-3, 2], [-2, 1])

    # Tensor grid parameters
    orders = [3,3,3,3]

    # Sparse grid parameters
    level = 2
    growth_rule = "linear"

    # Hyperbolic Cross parameters -- for effective quadratures
    q = 0.75

    # Get the integral approximations and the corresponding number of points
    sparse_grid_approx, sparse_grid_points = integrals.sparseGrid(parameter_ranges, level, growth_rule, function)
    tensor_grid_approx, tensor_grid_points = integrals.tensorGrid(parameter_ranges, orders, function)
    effectivequad_approx, effective_quad_points = integrals.effectivelySubsampledGrid(parameter_ranges, orders, q, function)

    # Show outputs!
    print '----INTEGRALS----'
    print 'Tensor grid approximation: '+'\t'+str(tensor_grid_approx)+'\t'+'with # of points:'+str(len(tensor_grid_points))
    print 'Sparse grid approximation: '+'\t'+str(sparse_grid_approx)+'\t'+'with # of points:'+str(len(sparse_grid_points))
    print 'Effectively subsampled grid approximation: '+'\t'+str(effectivequad_approx)+'\t'+'with # of points:'+str(len(effective_quad_points))

# Model or function!
def function(x):
    return np.cos(x[0]) + x[1]**2 + x[2]*x[3]

main()
