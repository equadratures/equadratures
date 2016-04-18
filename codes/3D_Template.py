#!/usr/bin/python
from PolyParams import PolynomialParam
from PolyParentFile import PolyParent
from IndexSets import IndexSet
import MatrixRoutines as matrix
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma



"""
    Optimal Quadrature Subsampling
    3D Template file for testing and debugging!

    Pranay Seshadri
    University of Cambridge
    ps583 <at> cam.ac.uk

"""
# Simple analytical function
def fun(x):
    return np.exp(x[:,0] + x[:,1] + x[:,2] )

def main():

    #--------------------------------------------------------------------------------------
    #
    #  USER'S NOTES:
    #        1. With the derivative flag on we recommend using 2X basis terms
    #        2. Input maximum number of permissible model evaluations
    #        3. Input number of points on the "full grid" (3x5 times number in line above)
    #
    #--------------------------------------------------------------------------------------
    highest_order = 50 # more for visualization
    derivative_flag = 0 # derivative flag on=1; off=0


    full_grid_points = 3 # full tensor grid
    min_value, max_value = -1, 1 # range of uncertainty --> assuming Legendre
    alpha_parameter, beta_parameter = 0, 0 # Jacobi polynomial values for Legendre

    # Uncertainty parameters
    uq_parameter1 = PolynomialParam("Jacobi", min_value, max_value, alpha_parameter, beta_parameter, derivative_flag, full_grid_points) # Setup uq_parameter
    uq_parameter2 = PolynomialParam("Jacobi", min_value, max_value, alpha_parameter, beta_parameter, derivative_flag, full_grid_points+1) # Setup uq_parameter
    uq_parameter3 = PolynomialParam("Jacobi", min_value, max_value, alpha_parameter, beta_parameter, derivative_flag, full_grid_points) # Setup uq_parameter
    uq_parameters = [uq_parameter1, uq_parameter2, uq_parameter3]

    # Compute elements of an index set:self, index_set_type, orders, level=None, growth_rule=None):
    indexset_configure = IndexSet("total order", [5,5,5])
    indices = IndexSet.getIndexSet(indexset_configure)

    # Create a PolyParent object!
    uq_structure = PolyParent(uq_parameters, indices)
    pts, wts = PolyParent.getTensorQuadrature(uq_structure)
    print('TENSOR GRID')
    print(pts, wts)
    #A = PolyParent.getMultivariateA(uq_structure, pts)
    #b = fun(pts)
    #x = matrix.solveLeastSquares(A, b) #-->Need to check this!!!!!

    """-------------------------------------------------------------------------

        - Randomized tensor grid
        - QR decomposition on randomized elements
        - Basis selection via total order / hyperbolic crosses

    -------------------------------------------------------------------------"""
    total_elements_in_tensor_grid = 3 * 4 * 3
    maximum_permitted_evals = total_elements_in_tensor_grid
    random_arrangement = np.random.permutation(total_elements_in_tensor_grid)
    random_subsamples = np.sort(random_arrangement[0:maximum_permitted_evals], axis=None)
    gauss_pts, gauss_wts = PolyParent.getRandomizedTensorGrid(uq_structure, random_subsamples)
    print('-----MY TEST------')
    print(gauss_pts, gauss_wts)
    #A = PolyParent.getMultivariateA(uq_structure, gauss_pts)
    #b = fun(gauss_pts)

    #print(x)

main()
