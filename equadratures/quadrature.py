"""The samples template."""
from equadratures.sampling_methods.montecarlo import Montecarlo
from equadratures.sampling_methods.tensorgrid import Tensorgrid
from equadratures.sampling_methods.sparsegrid import Sparsegrid
from equadratures.sampling_methods.userdefined import Userdefined
from equadratures.sampling_methods.induced import Induced
from equadratures.sampling_methods.sampling_template import Sampling
import numpy as np
class Quadrature(object):
    """ The class defines a Sampling object. It serves as a template for all sampling methodologies.

    Parameters
    ----------
    parameters : list 
        A list of parameters, where each element of the list is an instance of the Parameter class.
    basis : Basis 
        An instance of the Basis class corresponding to the multi-index set used.
    points : numpy.ndarray
        Array containing the quadrature points.
    mesh : numpy.ndarray
        Description needed.
    corr : numpy.ndarray, optional
        Description needed.
    oversampling : float, optional
        Description needed.
    """
    def __init__(self, parameters, basis, points, mesh, corr=None, oversampling=14.0):
        self.parameters = parameters
        self.basis = basis
        self.points = points
        self.mesh = mesh
        if self.mesh.lower() == 'tensor-grid' or self.mesh.lower() == 'univariate':
            self.samples = Tensorgrid(self.parameters, self.basis)
            self.list = None
        elif self.mesh.lower() == 'sparse-grid':
            self.samples = Sparsegrid(self.parameters, self.basis)
            self.list = self.samples.tensor_product_list
            self.sparse_weights = self.samples.sparse_weights
        elif self.mesh.lower() == 'monte-carlo':
            self.samples = Montecarlo(self.parameters, self.basis, corr, oversampling)
            self.list = None
        elif self.mesh.lower() == 'induced':
            self.samples = Induced(self.parameters, self.basis)
        elif self.mesh.lower() == 'user-defined':
            self.samples = Userdefined(self.parameters, self.basis, self.points)
        else:
            error_message()

    def get_points(self):
        """ Returns the quadrature points.

        Returns
        -------
        numpy.ndarray
            Array of quadrature points.
        """
        return self.samples.points

    def get_weights(self):
        """ Returns the quadrature weights.

        Returns
        -------
        numpy.ndarray
            Array of quadrature weights
        """
        return self.samples.weights

    def get_points_and_weights(self):
        """ Returns the quadrature points and weights.

        Returns
        -------
        tuple
            Tuple (points,weights) containing two numpy.ndarray's; the quadrature points and weights.
        """
        return self.samples.points, self.samples.weights

def error_message():
    raise ValueError( 'Oh no. Something went wrong in quadrature.py!')
