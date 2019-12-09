"""User defined samples."""
import numpy as np
from equadratures.sampling_methods.sampling_template import Sampling
class Userdefined(Sampling):
    """
    The class defines a Sampling object. It serves as a template for all sampling methodologies.

    :param list parameters: A list of parameters, where each element of the list is an instance of the Parameter class.
    :param Basis basis: An instance of the Basis class corresponding to the multi-index set used.
    """
    def __init__(self, parameters, basis, points):
        self.parameters = parameters
        self.basis = basis
        self.points = points
        super(Userdefined, self).__init__(self.parameters, self.basis, self.points)