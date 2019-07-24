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
        indices = Basis('Total order', tensor_sample)

        univar_induced_sampling = self.generate_sample_measure(self.param_type,
                                                               self.shape_parameter_A,
                                                               self.shape_parameter_B)
        x = self.inverse_mixture_sampling(self.dimension,
                                          self.sample_size,
                                          indices,
                                          univar_induced_sampling)

        poly = Poly(self.parameter, indices)
        polynomials = poly.getPolynomial(x)
        w = np.sum(polynomials**2, 0)

        return x, w

    def generate_sample_measure(self, param_type,
                                shape_parameter_A,
                                shape_parameter_B):

        """
        Generate the class of probability measures
        the input distribution belongs to.
        And return the univariate function object to used to sample
        """
        if param_type == "Beta":
            alpha = shape_parameter_B - 1.0
            beta = shape_parameter_A - 1.0
        if self.param_type == "Uniform":
            alpha = 0.0
            beta = 0.0
        if self.param_type == "Chebyshev":
            alpha = self.shape_parameter_B
            beta = self.shape_parameter_B

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
        indices.elements = indices.elements[sampled_indices_rows, :]

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
        # M = 10
        pass

        # return data
