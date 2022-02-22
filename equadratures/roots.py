import numpy as np

def colleague( c, parameter, tol=1E-12 ):
    """ Evaluates the roots of a Chebyshev polynomial with given coefficients, c.

    Training accuracy is evaluated on the data used for fitting the polynomial. Testing accuracy is evaluated on new data if it is provided by the ``X_test`` and ``y_test`` arguments (both must be provided together).

    Parameters
    ----------
    c : numpy.ndarray
        An ndarray with shape (number_of_coefficients, 1), containing coefficients of the modelled Chebyshev polynomial.
    parameter : :class:`~equadratures.parameter.Parameter`
        Contains the :class:`~equadratures.parameter.Parameter` object belonging to the polynomial.
    tol : float, optional
        An optional float for the machine tolerance.

    Returns
    -------
    numpy.ndarray
        Array of shape (number_of_roots,1) corresponding to the location the roots were found at.
    """
    c = c.ravel()
    order = parameter.order

    mat1 = parameter.get_jacobi_matrix(order)

    sset = c[:-1]
    sset *= 1/(2*c[-1])
    mat2 = np.zeros_like( mat1 )
    mat2[-1, :] = sset

    # Final colleague matrix
    C = mat1 - mat2

    # Compute eigenvalues of colleague matrix
    eigenvalues = np.linalg.eigvals(C)

    # Discard values with large imaginary parts
    roots = np.asarray(
            [eig.real for eig in eigenvalues
                if (eig.imag < tol) and
                    (np.abs(eig.real) <= 1.0 + tol)
            ]
    )

    # Ensure roots are within expected bounds of [-1.0, 1.0]
    roots = np.sort( roots[(roots > -1.0) & (roots < 1.0)] )

    return roots