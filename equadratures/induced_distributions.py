"""
Quadrature Numerical Integration Strategy
Based on The Induced Sampling Technique by Dr. Narayan
With the Christoffel Function based on Orthogonal Polynomials
With Optimal Sub-Sampling by Rank-Revealing QR

Classes
---------------------
OptimalSampling:
    Optimal sub-sampling from an Christoffel weighted random sampling

InducedSampling:
    Compute Samples from classes of polynomials induced probability measures
"""

from equadratures.parameter import Parameter
from equadratures.poly import Poly
from equadratures.basis import Basis

import numpy as np


class OptimalSampling:

    """
    This class would provide the optimal sampling strategy
    by sub-sampling a induced distribution
    with rank-revealing QR decomposition
    Providing an optimal quadrature based
    numerical integration pipeline

    :param Optimisation_method optimisation_method:
        type: String,
        Usage: Optimisation method for sub-sampling
        Example："greedy-qr" or "newton"
    """

    def __init__(self, optimisation_method):
        self.optimisation_method = optimisation_method


class InducedSampling:

    """
    This class is used for computing the samples from a
    polynomial induced distribution

    References for theory
    --------------------
    Seshadri, P., Iaccarino, G. and Tiziano, G.
    Quadrature Strategies for Constructing Polynomial Approximations,
    Springer
    2018

    Narayan, A.
    Computation of Induced Orthogonal Polynomial Distributions,
    numerical analysis ETNA
    2017

    Cohen, A. and Migliorati, G.
    Optimal Weighted Least Square Methods
    SMAI Journal of Computational Mathematics
    2017

    Parameters
    --------------------
    :param int order:
        type: int
        Usage: Specify the degree of polynomials used in approximation
        Example: degree = 3

    :param int dimension:
        type: int
        Usage: Specify the input dimension of the function to be approximated
        Example: dimension = 6

    :param int sample_size:
        type: int
        Usage: Specify number of samples for the quadrature estimation
        Example: sample_size = 60

    :param string param_type:
        type: String
        Usage: Specify type of probability measure used
        Example: "Uniform", "Beta"

    :param string shape_parameter_A:
        type: float
        Usage: The first shape parameter for a class of probability measures
        Example: alpha value for freud meaure or beta value for jacobi measure

    :param string shape_parameter_B:
        type: float
        Usage: The second shape parameter for a class of probability measures
        Example: rho value for freud meaure or alpha value for jacobi measure

    Public methods
    --------------------
    jacobi(self):
        :return:
            x: An array of sampled quadrature points(sample_size*dimension)
            w: Corresponding weights to each quadrature point(sample_size*1)

    freud(self):
        :return:
            x: An array of sampled quadrature points(sample_size*dimension)
            w: Corresponding weights to each quadrature point(sample_size*1)

    half_line_freud(self):
        :return:
            x: An array of sampled quadrature points(sample_size*dimension)
            w: Corresponding weights to each quadrature point(sample_size*1)
    """
    # Note recurrence coefficients

    def __init__(self, order,
                 dimension,
                 sample_size,
                 param_type,
                 shape_parameter_A=None,
                 shape_parameter_B=None):

        self.parameter = Parameter(order=order, distribution=param_type,
                                   shape_parameter_A=shape_parameter_A,
                                   shape_parameter_B=shape_parameter_B)
        self.sample_size = sample_size
        self.dimension = dimension
        self.order = order
        self.param_type = param_type
        self.shape_parameter_A = shape_parameter_A
        self.shape_parameter_B = shape_parameter_B

    def sample_quadrature_rule(self):
        """
        Performs induced Sampling on a Jacobi class of measures
        Produces the corresponding quadrature rule for integration

        :param Parameter self:
            utilises all initialised parameter in the self object
        :return:
            x: An array of sampled quadrature points(sample_size*dimension)
            w: Corresponding weights to each quadrature point(sample_size*1)
        """
        tensor_sample = [self.order]*self.dimension
        indices = IndexSet('Total order', tensor_sample)

        univar_induced_sampling = self.generate_sample_measure(self.param_type,
                                                               self.shape_parameter_A,
                                                               self.shape_parameter_B)
        x = self.inverse_mixture_sampling(self.dimension,
                                          self.sample_size,
                                          indices,
                                          univar_induced_sampling)

        poly = Poly(self.parameter, indices)
        polynomials = poly.getPolynomial(x)

    def generate_sample_measure(self, param_type,
                                shape_parameter_A,
                                shape_parameter_B):

        """
        Generate the class of probability measures
        the input distribution belongs to.
        And return the univariate function object to used to sample
        """
        if param_type is "Beta":
            alpha = shape_parameter_B - 1.0
            beta = shape_parameter_A - 1.0
        if self.param_type is "Uniform":
            alpha = 0.0
            beta = 0.0

        if param_type in ["Chebyshev", "Uniform", "Arcsine"]:
            return lambda cdf_values, indices:\
                self.inverse_induced_jacobi(cdf_values, indices, alpha, beta)

    @staticmethod
    def inverse_mixture_sampling(sample_size, dimension, indices, sampling_method):
        """
        Performs tensorial sampling from the additive mixture of induced distributions.

        :param int sample_size:
            number of sampled points returned
        :param int order
            number of sampled points returned
        :param int dimension
            number of sampled points returned
        :param Basis indices:
            Basis set of the tensorial indices for the polynomial variables
        :param self.functions sampling_method:
            The inverse induced sampling from this class
            Example: inverse_induced_jacobi(), inverse_induced_freud(),
            inverse_induced_hl_freud()

        :return:
        :param np.array x:
            A matrix x, of size (sample_size*dimension)
            specifying the sampled multi-variable input values
        """
        if sample_size <= 0 and type(sample_size) == int:
            raise ValueError("sample_size must be a positive integer")

        # TODO add a total order index set with random samples
        # The above would be necessary in higher dimensions
        indices_number = indices.elements.shape[0]
        sampled_indices_rows = np.ceil(indices_number * np.random.rand(sample_size, 1))
        indices.elements = indices.elements[sample_indices_rows, :]

        x = sampling_method(np.random.rand(sample_size, dimension), indices)

        return x

    def inverse_induced_jacobi(self):
        """
        Computations for a univariate inverse induced Jacobi distributions
        :param Parameter self:
            uses the self object initialised in the class
        :return:
            Returns a piecewise chebyshev interpolant coefficient as data
            of order self.order
        """
        # M-order quadrature for CDF estimation
        M = 10

        return data

    def induced_jacobi_distribution(self, x, n, M=None):
        """
        Evaluates the induced Jacobi distribution.
        :param Parameter self:
            An instance of the Parameter class.
        :param array x:
            Points over which the induced distribution must be computed.
        :param int order:
            Order of the distribution.
            Note that this value will override the order
            associated with the Parameter instance.
        :return:
            The median estimate (double)
        """
        if self.param_type is "Beta":
            alph = self.shape_parameter_B - 1.0  # bug fix @ 9/6/2016
            bet = self.shape_parameter_A - 1.0
        if self.param_type is "Uniform":
            alph = 0.0
            bet = 0.0
        if len(x) == 0:
            return
        assert((alph > -1) and (bet > -1))
        assert(all(np.abs(x[:]) <= 1))
        assert(n >= 0)
        return F

    def induced_distribution_jacobi_bisection(self, u, n, alpha, beta):
        """
        Computes the inverse of the order-n induced primitive
        for the Jacobi distribution
        with shape parameters alpha and beta.
        Uses a bisection method in conjunction with forward
        evaluation given by the induced jacobi distribution function.
        """
        assert((all(u) >= 0) and (all(u) <= 1))
        assert((alpha > -1) and (beta > -1))
        assert(n >= 0)
        x = np.zeros((len(u)))
        supp = [-1, 1]

        if n == 1:
            # TODO Consider not using a lambda expression here
            primitive = lambda x: self.induced_jacobi_distribution(x, n)
            ab = self.getRecurrenceCoefficients(2*n+400)
            x = self.inverse_distribution_primitive(u, n, primitive, supp)
        else:
            nmax = np.max(n)
            rr = np.arange(-0.5, 0.5+nmax, 1.)
            binvalues = np.digitize(n, rr)

        ab = self.getRecurrenceCoefficients(2*n+400)
        # x = idist_inverse!
        return 0

    def fast_induced_jacobi_distribution_setup_helper_1(ug, exps):
        N = len(ug)
        ug_mid = 0.5 * (ug[0:N-1] + ug[1:N])
        ug = np.append(ug, ug_mid)
        exponents = np.zeros((2, len(ug) - 1))

        for q in range(0, len(ug) - 1):
            if np.mod(q, 2) == 1:
                exponents[0, q] = 2.0/3.0
            else:
                exponents[1, q] = 2.0/3.0

        exponents[0, 0] = exps[0]
        exponents[1, N-1] = exps[1]
        return ug, exponents

    def fast_induced_jacobi_distribution_setup_helper_2(ug, idistinv, exponents, M):
        #xx = np.linspace(np.pi, 0, M+1)
        xx = np.linspace(0.5*np.pi, 0, M)
        vgrid = np.cos(xx)
        chebyparameter = Parameter(param_type='Chebyshev', order=M-1, lower=0.0, upper=1.0)
        V, __ = chebyparameter._getOrthoPoly(vgrid)
        iV = np.linalg.inv(V) # Shouldn't we replace this with a 
        lenug = len(ug) - 1
        ugrid = np.zeros((M, lenug))
        xgrid = np.zeros((M, lenug))
        xcoefficients = np.zeros((M, lenug))
        for q in range(0, lenug):
            ugrid[:,q] = (vgrid + 1.0) * 0.5 * ( ug[q+1] - ug[q] ) + ug[q]
            xgrid[:,q] = idistinv(ugrid[:,q])
            temp = xgrid[:,q]
            if exponents[0,q] != 0:
                temp = ( temp - xgrid[0,q] ) / (xgrid[lenug, q] - xgrid[0,q] )
            else:
                temp = ( temp - xgrid[0,q] ) / (xgrid[lenug, q] - xgrid[1, q] )
            
            for i in range(0, len(temp)):
                temp[i] = temp[i] * (1 + vgrid[i])**(exponents[0,q]) * (1 - vgrid[i])** exponents[1,q]
                if np.isinf(temp[i]) or np.isnan(temp[i]):
                    temp[i] = 0.0
            temp = np.reshape(temp, (M,1))
            xcoefficients[:,q] = np.reshape( np.dot(iV, temp), M)
    
        data = np.zeros((M + 6, lenug))
        for q in range(0, lenug):
            data[0,q] = ug[q]
            data[1,q] = ug[q+1]
            data[2,q] = xgrid[0,q]
            data[3,q] = xgrid[lenug,q]
            data[4,q] = exponents[0,q]
            data[5,q] = exponents[1,q]
            for r in range(6, lenug):
                data[r, q] = xcoefficients[r-6, q] 
        return data

    def median_approximation_jacobi(alpha, beta, n):
        """
        Returns an estimate for the median of the order-n Jacobi induced distribution.
        :param Parameter self:
            An instance of the Parameter class
        :param int order:
            Order of the distribution. Note that this value will override the order associated with the Parameter instance.
        :return:
            The median estimate (double)
        """
        if n > 0 :
            x0 = (beta**2 - alpha**2) / (2 * n + alpha + beta)**2
        else:
            x0 = 2.0/(1.0 + (alpha + 1.0)/(beta + 1.0))  - 1.0
        return x0

    def linearModification(ab, x0):
        """
        Performs a linear modification of the orthogonal polynomial recurrence coefficients. It transforms the coefficients
        such that the new coefficients are associated with a polynomial family that is orthonormal under the weight (x - x0)**2
        :param Parameter self:
            An instance of the Parameter class
        :param double:
            The shift in the weights
        :return:
            A N-by-2 matrix that contains the modified recurrence coefficients.
        """
      
        alpha = ab[:,0]
        length_alpha = len(alpha)
        beta = ab[:,1]
        sign_value = np.sign(alpha[0] - x0)
        r = np.reshape(np.abs(evaluateRatioSuccessiveOrthoPolynomials(alpha, beta, x0, N-1)) , (length_alpha - 1, 1) )
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
            ab[i,1] = beta[i] * bcorrect[i]
            ab[i, 0] = alpha[i] + sign * acorrect[i]
    
        return ab

    def quadraticModification(alphabeta, x0):
        """
        Performs a quadratic modification of the orthogonal polynomial recurrence coefficients. It transforms the coefficients
        such that the new coefficients are associated with a polynomial family that is orthonormal under the weight (x - x0)**2
        :param Parameter self:
            An instance of the Parameter class
        :param double:
            The shift in the weights
        :return:
            A N-by-2 matrix that contains the modified recurrence coefficients.
        """
        N = len(alphabeta)
        alpha = alphabeta[:,0]
        beta = alphabeta[:,1]
        C = np.reshape(  christoffelNormalizedOrthogonalPolynomials(alpha, beta, x0, N-1)  , [N, 1] )
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
            bcorrect[i] = (1.0 * temp1[i+1] ) / (1.0 *  temp1[i] )
        print bcorrect.shape
        print '-----*'
        bcorrect[0] = (1.0 + C[1]**2) * 1.0/(C[0]**2)
        for i in range(0, N-2):
            ab[i,1] = beta[i+1] * bcorrect[i]
            ab[i,0] = alpha[i+1] + acorrect[i]
        return ab

    def evaluateRatioSuccessiveOrthoPolynomials(a, b, x, N):
        # Evaluates the ratio between successive orthogonal polynomials!
        nx = len(x)
        assert (N>0), "This positive integer must be greater than 0!"
        assert (N < len(a)), "Positive integer N must be less than the number of elements in a!"
        assert (N < len(b)), "Positive integer N must be less than the number of elements in b!"
        r = np.zeros((nx, N))
    
        # Flatten x!
        xf = x[:]
        p0 = np.ones((nx, 1)) * 1.0/np.sqrt(b[0])
        p1 = np.ones((nx, 1))
        r1 = np.ones((nx, 1))
        r2 = np.ones((nx, 1))
        for i in range(0, nx):
            p1[i] = 1.0/np.sqrt(b[1]) * ( xf[i] - a[0] ) * p0[i]
            r1[i] = p1[i] / p0[i]
        r[:,0] = r1
    
        for q in range(1, N):
            for i in range(0, nx):
                r2[i] = ( xf[i] - a[q] ) - np.sqrt(b[q])/ r1[i]
                r1[i] = 1.0/np.sqrt(b[q+1]) * r2[i]
            r[:,q] = r1
    
        return r

    def christoffelNormalizedOrthogonalPolynomials(a, b, x, N):
        # Evaluates the Christoffel normalized orthogonal getPolynomialCoefficients
        nx = len(x)
        assert N>= 0
        assert N <= len(a)
        assert N <= len(b)
        C = np.zeros((nx, N+1))
        # Initialize the polynomials!
        C[:,0] = 1.0/ ( 1.0 * np.sqrt(b[0]) )
        if N > 0:
            for k in range(0, len(x)):
                C[k,1] = 1.0 / (1.0 * np.sqrt(b[1]) ) * (x[k] - a[0])
        if N > 1:
            for k in range(0, len(x)):
                C[k,2] = 1.0 / np.sqrt(1.0 + C[k,1]**2)  * (  (x[k] - a[1]) * C[k,1] - np.sqrt(b[1]) )
                C[k,2] = C[k,2] / (1.0 * np.sqrt(b[2]) )
        if N > 2:
            for nnn in range(2, N):
                for k in range(0, len(x)):
                    C[k,nnn+1] = 1.0/np.sqrt(1.0 + C[k,nnn]**2) * (  (x[k] - a[nnn]) * C[k,nnn] - np.sqrt(b[nnn]) * C[k,nnn-1] / np.sqrt(1.0 + C[k, nnn-1]**2) )
                    C[k,nnn+1] = C[k,nnn+1] / np.sqrt(b[nnn+1])
        return C
