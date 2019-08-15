"""Tensor grid based sampling."""
from equadratures.sampling_methods.sampling_template import Sampling
import numpy as np
class Tensorgrid(Sampling):
    """
    The class defines a Tensorgrid sampling object.

    :param list parameters: A list of parameters, where each element of the list is an instance of the Parameter class.
    :param Basis basis: An instance of the Basis class corresponding to the multi-index set used.
    """
    def __init__(self, parameters, basis, orders=None):
        self.parameters = parameters
        self.basis = basis
        if orders is not None:
            self.basis.set_orders(orders)
        self.dimensions = len(self.parameters)
        self.__set_points(orders)
        super(Tensorgrid, self).__init__(self.parameters, self.basis, self.points, self.weights)
    def __set_points(self, orders=None):
        """
        Generates a tensor grid quadrature rule based on the parameters in Poly.

        :param Poly self:
            An instance of the Poly class.
        :param list orders:
            A list of the highest polynomial orders along each dimension.
        """
        # Initialize points and weights
        pp = [1.0]
        ww = [1.0]

        if orders is None:
            orders = self.basis.orders

        # number of parameters
        # For loop across each dimension
        for u in range(0, self.dimensions):

            # Call to get local quadrature method (for dimension 'u')
            local_points, local_weights = self.parameters[u]._get_local_quadrature(orders[u])
            ww = np.kron(ww, local_weights)

            # Tensor product of the points
            dummy_vec = np.ones((len(local_points), 1))
            dummy_vec2 = np.ones((len(pp), 1))
            left_side = np.array(np.kron(pp, dummy_vec))
            right_side = np.array( np.kron(dummy_vec2, local_points) )
            pp = np.concatenate((left_side, right_side), axis = 1)

        # Ignore the first column of pp
        self.points = pp[:,1::]
        self.weights = ww
