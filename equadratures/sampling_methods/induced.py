from equadratures.sampling_methods.sampling_template import Sampling
import numpy as np
from scipy.special import betaln
import bisect
from scipy.optimize import brentq as brentq_root_solve


class Induced(Sampling):
    """
    The class defines an Induced sampling object.
    :param list parameters: A list of parameters,
                            where each element of the list is
                            an instance of the Parameter class.
    :param Basis basis: An instance of the Basis class
                        corresponding to the multi-index set used.
    """

    def __init__(self, parameters, basis, orders=None):
        self.parameters = parameters
        self.basis = basis
        if orders is not None:
            self.basis.set_orders(orders)
        else:
            orders = []
            for parameter in parameters:
                orders.append(parameter.order)
            self.basis.set_orders(orders)
        self.dimensions = len(self.parameters)
        self.basis_entries = basis.cardinality
        sampling_ratio = 5 * self.dimensions
        self.samples_number = int(sampling_ratio * self.basis.cardinality)
        self.sample_count = 0
        self.points = self._set_points(orders)
        self._set_weights()

    def _set_points(self, orders=None):
        """
        Performs induced sampling on
        Jacobi/Freud/Half-Line Freud classes of measures and
        produces the corresponding quadrature points for integration.
        :param Poly self:
            An instance of the Poly class.
        :param list orders:
            A list of the highest polynomial orders along each dimension.
        :returns Array quadrature_points:
            A vector of size (samples_number, dimension)
            of induced sampled quadrature points
        """

        # TODO add a total order index set with random samples
        # The above would be necessary in higher dimensions
        # Randomly sample index-set for each quadrature point
        indexset = self.basis.elements
        cardinality = self.basis.cardinality
        sampled_row_numbers = np.ceil(
                (cardinality-1)
                * np.random.rand(self.samples_number)
                ).astype(int)
        index_set_used = indexset[sampled_row_numbers]

        # Sample uniformly for inverse CDF
        sampled_cdf_values = np.random.rand(self.samples_number,
                                            self.dimensions)
        max_order = np.max(self.basis.orders)
        quadrature_points = self._additive_mixture_sampling(
                index_set_used,
                sampled_cdf_values,
                max_order
                )
        return quadrature_points

    def _additive_mixture_sampling(self, index_set, cdf_values, max_order):
        """
        perform sampling from
        an additive mixture of induced distributions
        of all orders
        by mapping each inverse induced cdf operation
        to the appropriate univariate polynomials

        hence perform a induced random sampling
        for each quadrature point.

        Assumes distributions in each dimension
        are independent
        and identical in the respective marginals

        Parameters:
        ---------
        index_set:
            a list total order index set used for each multi-variate sample
        Returns:
        ---------
        quadrature_points: np.ndarray
            A vector of size (samples_number, dimension)
            of induced sampled quadrature points
        """
        quadrature_points = np.zeros((self.samples_number,
                                     self.dimensions))
        parameter = self.parameters[0]
        for order in range(max_order):
            variable_positions = np.where(index_set == order)
            sampled_cdf_values = cdf_values[variable_positions]
            inverse_cdf_values = self._univariate_sampling(
                    parameter,
                    sampled_cdf_values,
                    order)
            quadrature_points[variable_positions] = inverse_cdf_values

        return quadrature_points

    def _univariate_sampling(self, parameter, cdf_values, order):
        """
        Generate the class of probability measures
        the input distribution belongs to.
        And return the univariate function object to used to sample
        Parameters
        ---------
        _input: tuple
            Internal variable passing
            a tuple of 3 elements
            1. probablity measure parameters of type Parameter
            2. the uniformly sampled cdf value for inverse cdf sampling
            of a type float between 0 and 1
            3. the degree of the univariate orthogonal polynomial
            of type int
        Returns
        ---------
        sampled_value: float
            a scalar value sampled
            according to the induced distribution
        """
        # TODO elaborate the following distributions to cover all param_types
        if order == 0.0:
            sampled_values = np.full(len(cdf_values), parameter.mean)
            return sampled_values
        if parameter.name in ["Uniform", "uniform"]:
            # TODO Only uniform between -1 and 1 is done for now
            # define shape parameters for the Jacobi matrix
            alpha = 0
            beta = 0
            parameter.order = order
            sampled_value = self.inverse_induced_jacobi(alpha,
                                                        beta,
                                                        cdf_values,
                                                        parameter)
            return sampled_value
        elif parameter.name in ["Gaussian"]:
            # TODO add Freud sampling algorithm
            pass
        pass

    def inverse_induced_jacobi(self, alpha, beta, cdf_values, parameter):
        """
        Sampling of a univariate inverse induced Jacobi distributions
        Parameters
        ---------
        parameter: Parameter
            Parameter class to specify the ditribution parametersA
        uniform_cdf_value: float
            A floating point number between 0 and 1
            for the inverse CDF sampling method
        order:
            The order of the induced polynomial
        Returns:
        ---------
        sampled_value: ndarray
            a vector value sampled of shape (len(cdf_values))
            according to the induced distribution
        """
        # TODO use Chebyshev interpolants for the induced distributions
        # The method above pre-evaluates the CDF
        # Hence gives efficient subsequent evaluation so to speak

        # Use Markov-Stiltjies inequality for initial x value interval guess
        order = int(parameter.order)
        zeroes, _ = parameter._get_local_quadrature(order-1)
        # obtain current recurrence coefficient
        ab = parameter.get_recurrence_coefficients((order)*2 + 400-1)
        for root in zeroes:
            ab = self._quadratic_modification(ab, root)
            ab[0, 1] = 1
        induced_points, induced_weights =\
            parameter._get_local_quadrature(400-1, ab)
        # insert lower bound of x in jacobi distribution
        interval_points = np.insert(induced_points, 0, -1)
        # Cumulative sums of induced quadrature weights
        # are a strict bound for the cdf
        strict_bounds = np.cumsum(induced_weights)
        strict_bounds = np.insert(strict_bounds, len(strict_bounds), 1)
        strict_bounds = np.insert(strict_bounds, 0, 0)
        sampled_values = np.zeros(len(cdf_values))
        # Solver function for inverse CDF where F(x)-u = 0
        def F(x, CDFVAL):
            value = self.induced_jacobi_evaluation(alpha,
                                                   beta,
                                                   x,
                                                   parameter)
            value = value - CDFVAL
            return value
        sample = 0
        for cdf_value in cdf_values:
            interval_index = bisect.bisect_left(strict_bounds, cdf_value)
            interval_index_hi = interval_index
            if interval_index_hi >= 399:
                interval_lo = interval_points[interval_index-1]
                interval_hi = 1
            else:
                interval_lo = interval_points[interval_index-1]
                interval_hi = interval_points[interval_index_hi+1]
            CDFVAL = cdf_value
            sampled_value = brentq_root_solve(F, interval_lo,
                                              interval_hi,
                                              (CDFVAL),
                                              xtol=0.00005)
            sampled_values[sample] = sampled_value
            sample += 1

        return sampled_value

    def induced_jacobi_evaluation(self, alpha, beta, x, parameter):
        """
        Evaluate induced Jacobi distribution CDF value
        Parameters
        ---------
        alpha: double
            alpha shape parameter
            of the jacobi distribution
        beta: double
            beta shape parameter
            of the jacobi distribution
        x: double
            input variable of the distribution
        parameter: Parameter
            the parameter class for the distribution
        Returns
        ---------
        F: double
            The CDF value evaluated at x
        """
        # M-order quadrature for CDF estimation
        _complementary = False
        M = 12
        order = parameter.order
        # Avoid division by zero and simplify computation at bounds
        if int(x) == 1:
            F = 1
            return F
        if x == -1:
            F = 0
            return F
        # find median by the Mhaskar-Rakhmanov-Saff numbers
        if order > 0:
            median = (beta**2 - alpha**2)/((2*order + alpha + beta)**2)
        else:
            median = 2.0/(1.0 + (alpha + 1.0)/(beta + 1.0)) - 1.0
        # Set up computation for the complementary value
        if x > median:
            x = -x
            _complementary = True
            alpha, beta = beta, alpha
            parameter.shape_parameter_A, parameter.shape_parameter_B = \
                parameter.shape_parameter_B, parameter.shape_parameter_A

        # Obtain the zeroes of this particlar polynomial
        zeroes, _ = parameter._get_local_quadrature(order-1)
        ab = parameter.get_recurrence_coefficients(order)

        # This is the (inverse) n'th root of the leading coefficient square of p_n
        # We'll use it for scaling later
        scaling_kn_factor = np.exp(-1.0/order
                                   * np.sum(np.log(ab[:, 1])))

        # Recurrence coefficients for the quadrature rule
        A = np.floor(abs(alpha))
        recurrence_ab = parameter.get_recurrence_coefficients(2*order+A+M)
        logfactor = 0.0  # factor to keep the modified distribution as a pdf

        # n quadratic modifications
        # for the induced distribution
        # recurrence coefficients
        for i in range(0, int(order)):
            quadratic_root = (2.0/(x+1.0)) * (zeroes[i] + 1.0) - 1.0
            recurrence_ab = self._quadratic_modification(recurrence_ab,
                                                          quadratic_root)
            logfactor += np.log(recurrence_ab[0, 1] *
                                ((x+1.0)/2.0)**2 *
                                scaling_kn_factor)
            recurrence_ab[0, 1] = 1

        linear_root = (3-x)/(1+x)

        # A quadratic modifications
        # for the induced distribution
        # recurrence coefficients
        for j in range(0, int(A)):
            recurrence_ab = self._linear_modification(recurrence_ab,
                                                       linear_root)
            logfactor += logfactor + np.log(ab[0, 1] *
                                            1.0/2.0 *
                                            (x+1.0))
            recurrence_ab[0, 1] = 1
        u, w = parameter._get_local_quadrature(M-1, recurrence_ab)
        integral = np.dot(w, (2.0 - 1.0/2.0 * (u+1.) * (x+1.))**(alpha-A))

        F = np.exp(logfactor -
                   alpha*np.log(2.0) -
                   betaln(beta+1.0, alpha+1.0) -
                   np.log(beta+1.0)+(beta+1)*np.log((x+1.0)/2.0))*integral
        F = np.asscalar(F)

        if _complementary:
            F = 1-F

        return F

    def _linear_modification(self, ab, x0):
        """
        Performs a linear modification of the
        orthogonal polynomial recurrence coefficients.
        It transforms the coefficients
        such that the new coefficients
        are associated with a polynomial family
        that is orthonormal under the weight (x - x0)**2
        Parameters
        ---------
        ab: np.ndarray
            numpy array of the alpha beta recurrence coefficients
        x0: float
            The shift in the weights
        Returns
        ---------
        A N-by-2 matrix that contains the modified recurrence coefficients.
        """

        alpha = ab[:, 0]
        N = len(alpha)
        beta = ab[:, 1]
        sign_value = np.sign(alpha[0] - x0)

        r = np.reshape(np.abs(self._polynomial_ratios(alpha,
                                                     beta, x0, N-1)),
                       (N-1, 1))

        acorrect = np.zeros((N-1, 1))
        bcorrect = np.zeros((N-1, 1))
        ab = np.zeros((N-1, N-1))

        for i in range(0, N-1):
            acorrect[i] = np.sqrt(beta[i+1]) * 1.0 / r[i]
            bcorrect[i] = np.sqrt(beta[i+1]) * r[i]

        for i in range(1, N-1):
            acorrect[i] = acorrect[i+1] - acorrect[i]
            bcorrect[i] = bcorrect[i] * 1.0/bcorrect[i-1]

        for i in range(0, N-1):
            ab[i, 1] = beta[i] * bcorrect[i]
            ab[i, 0] = alpha[i] + sign_value * acorrect[i]

        return ab

    def _quadratic_modification(self, alphabeta, x0):
        """
        Performs a linear modification of the
        orthogonal polynomial recurrence coefficients.
        It transforms the coefficients
        such that the new coefficients
        are associated with a polynomial family
        that is orthonormal under the weight (x - x0)**2
        Parameters
        ---------
        ab: np.ndarray
            numpy array of the alpha beta recurrence coefficients
        x0: float
            The shift in the weights
        Returns
        ---------
        A N-by-2 matrix that contains the modified recurrence coefficients.
        """
        N = len(alphabeta)
        alpha = alphabeta[:, 0]
        beta = alphabeta[:, 1]

        C = np.reshape(self._christoffel_normalised_polynomials(alpha,
                                                                 beta,
                                                                 x0,
                                                                 (N-1)),
                       [N, 1])
        acorrect = np.zeros((N-2, 1))
        bcorrect = np.zeros((N-2, 1))
        ab = np.zeros((N-2, 2))
        temp1 = np.zeros((N-1, 1))
        for i in range(0, N-1):
            temp1[i] = np.sqrt(beta[i+1]) * C[i+1] * C[i] * 1.0/np.sqrt(1.0 + C[i]**2)
        temp1[0] = np.sqrt(beta[1])*C[1]
        acorrect = np.diff(temp1, axis=0)
        temp1 = 1 + C[0:N-1]**2
        for i in range(0, N-2):
            bcorrect[i] = (1.0 * temp1[i+1]) / (1.0*temp1[i])
        bcorrect[0] = (1.0 + C[1]**2) * 1.0/(C[0]**2)
        for i in range(0, N-2):
            ab[i, 1] = beta[i+1] * bcorrect[i]
            ab[i, 0] = alpha[i+1] + acorrect[i]
        return ab

    def _polynomial_ratios(self, a, b, x, N):
        """
        Evaluates the ratio between successive orthogonal polynomials
        This is a helper method to perform the linear modification
        of the induced distribution.
        Parameters
        ---------
        a: np.ndarray
        alpha recurrence coefficients,
        size(N+1, 1)
        b: np.ndarray
        beta recurrence coefficients,
        size(N+1, 1)
        x: double
        a scalar double value of the evaluated input
        for the induced distribution
        N: int
        Order of the polynomials to be evaluated
        Returns
        ---------
        r: np.ndarray
        successive ratios in a vetor
        of length N
        """
        if N > 0:
            raise ValueError("number of polynomial ratios must be greater than 0")
        if N < len(a):
            raise ValueError("number of polynomial ratios must be less than length of a")
        if N < len(b):
            raise ValueError("number of polynomial ratios must be less than length of b")

        r = np.zeros(N)

        # find p0 and p1 to initialise r
        p0 = 1.0/np.sqrt(b[0])
        p1 = 1.0/np.sqrt(b[1]) * (x - a[0]) * p0
        r1 = p1 / p0
        r[0] = r1

        for q in range(1, N):
            r2 = (x - a[q]) - np.sqrt(b[q])/r1
            r1 = 1.0/np.sqrt(b[q+1]) * r2
            r[q] = r1

        return r

    def _christoffel_normalised_polynomials(self, a, b, x, N):
        """
        Computes the Christoffel normalized
        orthogonal polynomial values
        at x
        Parameters
        ---------
        a: np.ndarray
        alpha recurrence coefficients,
        size(N+1, 1)
        b: np.ndarray
        beta recurrence coefficients,
        size(N+1, 1)
        x: double
        a scalar double value of the evaluated input
        for the induced distribution
        N: int
        Order of the polynomials to be evaluated
        Returns
        ---------
        C: np.ndarray
        successive ratios in a vetor
        of length N
        """
        if N <= 0:
            raise ValueError("No. of Christoffels evaluations must be greater than 0")
        if N > len(a):
            raise ValueError("No. of Christoffels evaluations must be less than len(a)")
        if N > len(b):
            raise ValueError("No. of Christoffels evaluations must be less than len(b)")

        C = np.zeros(N+1)
        # Initialize the polynomials
        C[0] = 1.0/(1.0 * np.sqrt(b[0]))
        if N > 0:
            C[1] = 1.0 / (1.0 * np.sqrt(b[1])) * (x - a[0])
        if N > 1:
            C[2] = 1.0 / np.sqrt(1.0 + C[1]**2) * ((x - a[1]) * C[1] - np.sqrt(b[1]))
            C[2] = C[2] / (1.0 * np.sqrt(b[2]))
        if N > 2:
            for nnn in range(2, N):
                C[nnn+1] = 1.0/np.sqrt(1.0 + C[nnn]**2) * \
                    (
                     (x - a[nnn])*C[nnn] - np.sqrt(b[nnn]) *
                     C[nnn-1]/np.sqrt(1.0+C[nnn-1]**2)
                     )
                C[nnn+1] = C[nnn+1] / np.sqrt(b[nnn+1])
        return C
