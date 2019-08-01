"""The samples template."""
from equadratures.sampling_methods.montecarlo import Montecarlo
from equadratures.sampling_methods.tensorgrid import Tensorgrid
from equadratures.sampling_methods.sparsegrid import Sparsegrid
import numpy as np
class Quadrature(object):
    """
    The class defines a Sampling object. It serves as a template for all sampling methodologies.

    :param list parameters: A list of parameters, where each element of the list is an instance of the Parameter class.
    :param Basis basis: An instance of the Basis class corresponding to the multi-index set used.
    """
    def __init__(self, parameters, basis, points, outputs, correlation, mesh):
        self.parameters = parameters
        self.basis = basis
        self.points = points
        self.weights = weights
        self.correlation = correlation
        self.mesh = mesh
        choices = {'tensor-grid': Tensorgrid(self.parameters, self.basis),\
        'sparse-grid': Sparsegrid(self.parameters, self.basis), \
        'monte-carlo': Montecarlo(self.parameters, self.basis)}
        self.sampling = choices.get(self.name.lower(), error_message)
    def get_points(self):
        """
        Returns the quadrature points.

        :param Sampling self:
                An instance of the sampling class.
        """
        return self.sampling.points
    def get_weights(self):
        """
        Returns the quadrature weights.

        :param Sampling self:
                An instance of the sampling class.
        """
        return self.sampling.weights
    def get_points_and_weights(self):
        """
        Returns the quadrature points and weights.

        :param Sampling self:
                An instance of the sampling class.
        """
        return self.sampling.points, self.sampling.weights
def error_message():
    raise(ValueError, 'Oh no. Something went wrong in samples.py!')