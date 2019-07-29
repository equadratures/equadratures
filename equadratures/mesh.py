"""The mesh class."""
from equadratures.parameter import Parameter
from equadratures.basis import Basis
import numpy as np
class Mesh(Poly):
    """
    Generates a mesh.
    """
    def __init__(self, parameters, basis, method, samples):
        super(Poly, self).__init__(parameters, basis, method, samples)

    def get_sparse_grid_quadrature(self):

    def get_monte_carlo_quadrature(self):

    def get_tensor_grid_quadrature(self, orders=None):
        """
        Generates a tensor grid quadrature rule based on the parameters in Poly.
        :param Poly self:
            An instance of the Poly class.
        :param list orders:
            A list of the highest polynomial orders along each dimension.
        :return:
            A numpy array of quadrature points.
        :return:
            A numpy array of quadrature weights.
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
        points = pp[:,1::]
        weights = ww

        # Return tensor grid quad-points and weights
        return points, weights
