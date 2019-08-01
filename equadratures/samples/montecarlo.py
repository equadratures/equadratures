"""Monte Carlo based sampling."""
import numpy as np
from equadratures.samples.sampling_template import Sampling
CONST = 5
class Montecarlo(Sampling):
    """
    The class defines a Sampling object. It serves as a template for all sampling methodologies.

    :param list parameters: A list of parameters, where each element of the list is an instance of the Parameter class.
    :param Basis basis: An instance of the Basis class corresponding to the multi-index set used.
    """
    def __init__(self, parameters=None, basis=None):
        self.parameters = parameters
        self.basis = basis
        self.__set_points()
        super(Sampling, self).__init__(parameters, basis)
    def __set_points(self, number_of_samples= int(self.basis.cardinality * self.dimensions * CONST) ):
        """
        Sets the quadrature points and weights.

        :param Sampling self:
                An instance of the sampling class.
        """
        self.points = np.zeros((number_of_samples, self.dimensions))
        for i in range(0, self.dimensions):
            univariate_samples = self.parameters[i].getSamples(m_big)
            for j in range(0, m_big):
                self.points[j, i] = univariate_samples[j]
