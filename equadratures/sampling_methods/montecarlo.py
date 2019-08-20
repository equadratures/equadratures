"""Monte Carlo based sampling."""
import numpy as np
from equadratures.sampling_methods.sampling_template import Sampling
CONST = 5
class Montecarlo(Sampling):
    """
    The class defines a Sampling object. It serves as a template for all sampling methodologies.

    :param list parameters: A list of parameters, where each element of the list is an instance of the Parameter class.
    :param Basis basis: An instance of the Basis class corresponding to the multi-index set used.
    """
    def __init__(self, parameters, basis):
        self.parameters = parameters
        self.basis = basis
        self.dimensions = len(self.parameters)
        number_of_samples = int(self.basis.cardinality * len(self.parameters) * CONST)
        self.points = self._set_points(number_of_samples)
        super(Montecarlo, self).__init__(self.parameters, self.basis, self.points)

    def _set_points(self, number_of_samples):
        """
        Sets the quadrature points and weights.

        :param Sampling self:
                An instance of the sampling class.
        """
        self.points = np.zeros((number_of_samples, self.dimensions))
        for i in range(0, self.dimensions):
            univariate_samples = self.parameters[i].get_samples(number_of_samples)
            for j in range(0, number_of_samples):
                self.points[j, i] = univariate_samples[j]
        return self.points
