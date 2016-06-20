

# Replace old figures 3 and 4 with new ones in 2D!!!


# Then create the A matrix using a subset of columns
# Compute A and C matrices and solve the full least squares problem
A, C, gaussPoints = PolynomialParam.getAmatrix(uq_parameter1)
b = fun(gaussPoints, derivative_flag, error_flag)

# Normalize these!
Aweighted , NormFactor = matrix.rowNormalize(A)
bweighted = np.dot(NormFactor, b)

# "REAL" solution
x_true = matrix.solveLeastSquares(Aweighted, bweighted)

# Get the function values at ALL points!
function_values = fun(gaussPoints, derivative_flag, error_flag)

for basis_subsamples in range(2,highest_order):
    for quadrature_subsamples in range(2,highest_order):

        # Now compute the "optimal" subsamples from this grid!
        P = matrix.QRColumnPivoting( A[:, 0 : quadrature_subsamples] )
        optimal = P[ 0 : quadrature_subsamples]

        # Now take the first "evaluations_user_can_afford" rows from P
        Asquare = A[optimal, 0 : basis_subsamples]
        bsquare = b[optimal]
        rows, cols = Asquare.shape

        # Normalize these!
        Asquare, smallNormFactor = matrix.rowNormalize(Asquare)
        bsquare = np.dot(smallNormFactor, bsquare)


        # Solve least squares problem only if rank is not degenrate!
        if(np.linalg.matrix_rank(Asquare) == cols):
            # Solve the least squares problem
            x = matrix.solveLeastSquares(Asquare, bsquare)
            store_error[basis_subsamples,quadrature_subsamples] = np.linalg.norm( x - x_true[0:basis_subsamples])

            # Compute the condition numbers of these matrices!
            store_cond[basis_subsamples, quadrature_subsamples] = np.linalg.cond(Asquare)
