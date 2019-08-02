"""The samples template."""
from equadratures.sampling_methods.montecarlo import Montecarlo
from equadratures.sampling_methods.tensorgrid import Tensorgrid
from equadratures.sampling_methods.sparsegrid import Sparsegrid
from equadratures.sampling_methods.sampling_template import Sampling
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
        self.correlation = correlation
        self.mesh = mesh
        if self.mesh == 'tensor-grid':
            self.samples = Tensorgrid(self.parameters, self.basis)
        elif self.mesh == 'sparse-grid':
            self.samples = Sparsegrid(self.parameters, self.basis)
            self.solver = lambda model_evals: self.samples.__get_spam_solver(model_evals)
        elif self.mesh == 'monte-carlo':
            self.samples = Montecarlo(self.parameters, self.basis)
        else:
            error_message()
    def get_points(self):
        """
        Returns the quadrature points.

        :param Sampling self:
                An instance of the sampling class.
        """
        return self.samples.points
    def get_weights(self):
        """
        Returns the quadrature weights.

        :param Sampling self:
                An instance of the sampling class.
        """
        return self.samples.weights
    def get_points_and_weights(self):
        """
        Returns the quadrature points and weights.

        :param Sampling self:
                An instance of the sampling class.
        """
        return self.samples.points, self.samples.weights
def error_message():
    raise(ValueError, 'Oh no. Something went wrong in samples.py!')