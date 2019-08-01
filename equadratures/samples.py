"""The samples template."""
import numpy as np
from equadratures.samples.montecarlo import Montecarlo
from equadratures.samples.tensorgrid import Tensorgrid
from equadratures.samples.sparsegrid import Sparsegrid
class Samples(object):
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
        choices = {'tensor-grid': Tensorgrid(self.parameters, self.basis),
			       'sparse-grid': Sparsegrid(self.parameters, self.basis),
                   'monte-carlo': Montecarlo(self.parameters, self.basis),
				   }
		distribution = choices.get(self.name.lower(), distributionError)
		self.distribution = distribution
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


 Now call the relevant meshes!
        if self.mesh == 'tensor-grid':
            self.sampling = Tensorgrid(self.parameters, self.basis)
        elif self.mesh == 'sparse-grid':
            self.sampling = Sparsegrid(self.parameters, self.basis)
        elif self.mesh == 'monte-carlo':
            self.sampling = Montecarlo(self.parameters, self.basis)
        elif self.mesh == 'induced':
            self.sampling = Induced(self.parameters, self.basis)