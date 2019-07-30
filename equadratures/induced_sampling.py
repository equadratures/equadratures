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
import numpy as np
# from scipy.special import beta


class OptimalSubSampling:

    """
    This class would provide the optimal sampling strategy
    by sub-sampling a induced distribution
    with rank-revealing QR decomposition
    Providing an optimal quadrature based
    numerical integration pipeline

    Parameters
    ---------
    optimisation_method: String
        type: String,
        Usage: Optimisation method for sub-sampling
        Example: "greedy-qr" or "newton"
    """

    def __init__(self, optimisation_method):
        self.optimisation_method = optimisation_method


class InducedSampling:

    """
    This class is used for computing the samples from a
    polynomial induced distribution

    References for theory
    ---------
    Seshadri, P., Iaccarino, G. and Ghisu, T.
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
    ---------

    parameters : list
        A list of parameters,
        each element of the list is
        an instance of the Parameter class.

    basis : Basis
        An instance of the Basis class
        corresponding to the multi-index set used.

    sampling_ratio : double
        The ratio of the number of samples
        to the number of coefficients (cardinality of the basis).
        Should be greater than 1.0 for 'least-squares'.

    subsampling_method: String
        The type of subsampling required.
        In the aforementioned four sampling strategies,
        we generate a logarithm factor of samples above the required amount
        and prune down the samples using an optimisation technique.

        Avaliable options include:
            - 'qr'
            - 'lu'
            - 'svd'
            - 'newton'


    Public methods
    ---------
    samples(self):
        Returns:
        ---------
        x: An array of sampled quadrature points(sample_size*dimension)
    """
    # Note recurrence coefficients

    def __init__(self, parameters, basis, sampling_ratio, subsampling_method):

        self.parameters = parameters
        self.basis = basis
        self.subsampling = OptimalSubSampling(subsampling_method)
        self.dimension = len(parameters)
        self.samples_number = sampling_ratio * self.dimension

    def samples(self):
        """
        Performs induced Sampling on Jacobi/Freud/Half-Line Freud
        classes of measures
        Produces the corresponding quadrature points for integration
        Parameters:
        ---------
        self: Self
            utilises the parameters, basis arguments initialised in the class's state

        Returns:
        ---------
        quadrature_points: np.ndarray
            An array of sampled quadrature points(sample_size*dimension)
        """

        quadrature_points = np.zeros((self.samples_number, self.dimension))
        quadrature_points = np.apply_along_axis(self.additive_mixture_sampling,
                                                1, quadrature_points)

        return quadrature_points

    def additive_mixture_sampling(self, _placeholder):
        """
        Performs tensorial sampling from the additive mixture of induced distributions.

        Parameters:
        ---------
        _placeholder: np.ndarray
            This is an internal variable for the np.apply_along_axis function
            an array of size (dimension*1)

        Returns:
        ---------
        x: np.ndarray
            A vector x, of size (dimension*1)
            A single quadrature point sampled from
            the additive mixture of the induced distributions
        """
        # TODO add a total order index set with random samples
        # The above would be necessary in higher dimensions
        # sample the set of indices used in this sample
        indexset = self.basis.elements
        sampled_row_number = np.random.randint(0, indexset.shape[0])
        index_set_used = indexset[sampled_row_number, :]

        # Sample uniformly for inverse CDF
        sampled_cdf_values = np.random.rand(self.dimension, 1)

        x = self.multi_variate_sampling(sampled_cdf_values, index_set_used)

        return x

    def multi_variate_sampling(self, sampled_cdf_values, index_set_used):
        """
        Sample each dimension of the input vairable with their individual
        1. probablity measure parameters
        2. the uniformly sampled cdf value for inverse cdf sampling
        3. the degree of the univariate orthogonal polynomial

        Parameters
        ---------
        sampled_cdf_values: np.ndarray
            A list of uniformly generated CDF values
            for sampling in each dimension
            array size (dimension*1)

        index_set_used: np.ndarray
            The order of each univariate orthogonal polynomial
            used in each univariate induced dittributions
            array size(dimension*1)

        Returns
        ---------
        x: np.ndarray
            A vector x, of size (dimension*1)
            A single quadrature point sampled from
            the additive mixture of the induced distributions
        """
        univariate_input = zip(self.parameters, sampled_cdf_values, index_set_used)
        x = np.fromiter(map(self.univariate_sampling, univariate_input), float)

        return x

    def univariate_sampling(self, _input):
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
        parameter = _input[0]
        uniform_cdf_value = np.asscalar(_input[1])
        order = np.asscalar(_input[2])
        # TODO elaborate the following distribution types to cover all param_types
        if parameter.param_type in ["Uniform", "Chebyshev"]:
            sampled_value = self.inverse_induced_jacobi(parameter,
                                                        uniform_cdf_value,
                                                        order)
            return sampled_value
        elif parameter.param_type in ["Gaussian"]:
            # TODO add Freud sampling algorithm
            pass
        pass

    @staticmethod
    def inverse_induced_jacobi(parameter, uniform_cdf_value, order):
        """
        Sampling of a univariate inverse induced Jacobi distributions
        :param Parameter self:
            uses the self object initialised in the class
        :return:
            Returns a piecewise chebyshev interpolant coefficient as data
            of order self.order
        """
        # M-order quadrature for CDF estimation
        # M = 10
        pass
