"""Monte Carlo based sampling."""
import numpy as np
from equadratures.sampling_methods.sampling_template import Sampling
class Montecarlo(Sampling):
    """
    The class defines a Montecarlo object. Samples are generated from the given distribution.

    :param list parameters: A list of parameters, where each element of the list is an instance of the Parameter class.
    :param Basis basis: An instance of the Basis class corresponding to the multi-index set used.
    :param Correlations corr: An instance of Correlations object if input is correlated.
    """
    def __init__(self, parameters, basis, corr=None, oversampling=7.0):
        self.parameters = parameters
        self.basis = basis
        self.dimensions = len(self.parameters)
        number_of_samples = int(self.basis.cardinality * oversampling)
        self.points = self._set_points(number_of_samples, corr)
        super(Montecarlo, self).__init__(self.parameters, self.basis, self.points)

    def _set_points(self, number_of_samples, corr=None):
        """
        Sets the quadrature points and weights.

        :param Sampling self:
                An instance of the sampling class.
        """
        if not(corr is None):
            self.points = corr.get_correlated_samples(number_of_samples)
        else:
            self.points = np.zeros((number_of_samples, self.dimensions))
            for i in range(0, self.dimensions):
                univariate_samples = self.parameters[i].get_samples(number_of_samples)
                self.points[:, i] = univariate_samples[:]

        return self.points
