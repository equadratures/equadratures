#!/usr/bin/python
import PolyUsers as poly
from PolyParams import PolynomialParam
import MatrixRoutines as matrix
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
"""
    Optimal Quadrature Subsampling
    1D Example

    Pranay Seshadri
    University of Cambridge
    ps583 <at> cam.ac.uk

"""

# Simple analytical function
def fun(x):
    return np.exp(x[:])


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


    full_grid_points = 100 # full tensor grid
    min_value, max_value = -1, 1 # range of uncertainty --> assuming Legendre
    alpha_parameter, beta_parameter = 0, 0 # Jacobi polynomial values for Legendre
    uq_parameter1 = PolynomialParam("Jacobi", min_value, max_value, alpha_parameter, beta_parameter, derivative_flag, full_grid_points) # Setup uq_parameter

    # Compute A and C matrices!
    A, C, gaussPoints = PolynomialParam.getAmatrix(uq_parameter1)
    b = fun(gaussPoints)

    # Compute the full error!
    x_true = matrix.solveLeastSquares(A, b)


    # Pick select columns. This amounts using either a total order or hyperbolic cross
    # basis set in nD
    store_error = np.zeros((highest_order, highest_order)) + np.NaN
    for basis_subsamples in range(2,highest_order):
        for quadrature_subsamples in range(2,highest_order):

            # Subselect columns
            Atall = A[:, 0 : basis_subsamples]

            # Now compute the "optimal" subsamples from this grid!
            P = matrix.QRColumnPivoting(Atall)

            # To avoid rank degenracy let the number of rows be atleast equal to, if
            # not greater than the number of columns!
            if(quadrature_subsamples >= basis_subsamples):

                # Pick selet subsamples
                P = P[ 0 : quadrature_subsamples]

                # Now take the first "evaluations_user_can_afford" rows from P
                Asquare = Atall[P,:]

                # Row normalize the matrix!
                Asquare_norms = np.sqrt(np.sum(Asquare**2, axis=1)/(1.0 * quadrature_subsamples))
                Normalization_Constant = np.diag(1.0/Asquare_norms)
                A_LSQ = np.dot(Normalization_Constant, Asquare)
                b_LSQ = np.dot(Normalization_Constant, fun(gaussPoints[P,:] )  )

                # Solve the least squares problem
                x = matrix.solveLeastSquares(A_LSQ, b_LSQ)
                store_error[basis_subsamples,quadrature_subsamples] = np.linalg.norm( x - x_true[0:basis_subsamples])

    # Plot!
    Zm = ma.masked_where(np.isnan(store_error),store_error)
    yy, xx = np.mgrid[0:highest_order, 0: highest_order]
    plt.pcolor(yy, xx, np.log10(np.abs(Zm)), cmap='jet', vmin=-15, vmax=0)
    plt.colorbar()
    plt.ylabel('Quadrature subsamples')
    plt.xlabel('Basis subsamples')
    plt.title('L2 NORM COEFFICIENT ERROR')
    plt.xlim(2, highest_order-1)
    plt.ylim(2, highest_order-1)
    plt.show()

main()
