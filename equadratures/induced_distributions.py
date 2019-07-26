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


class OptimalSampling:

    """
    This class would provide the optimal sampling strategy
    by sub-sampling a induced distribution
    with rank-revealing QR decomposition
    Providing an optimal quadrature based
    numerical integration pipeline

    :param String optimisation_method:
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

    subsampling: String
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
    --------------------
    samples(self):
        Returns:
        ---------
        x: An array of sampled quadrature points(sample_size*dimension)
    """
    # Note recurrence coefficients

    def __init__(self, parameters, basis, sampling_ratio, subsampling):

        self.parameters = parameters
        self.basis = basis
        self.sampling_ratio = sampling_ratio
        self.subsampling = subsampling

    def samples(self):
        """
        Performs induced Sampling on a Jacobi class of measures
        Produces the corresponding quadrature rule for integration
        Parameters:
        ---------
        self: Self
            utilises the parameters, basis arguments initialised in the class's state

        Returns:
        ---------
            x: An array of sampled quadrature points(sample_size*dimension)
        """
        pass

    def generate_sample_measure(self):

        """
        Generate the class of probability measures
        the input distribution belongs to.
        And return the univariate function object to used to sample
        """
        pass

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

    def inverse_induced_jacobi(self, n):
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
