"""Tensor grid based sampling."""
from equadratures.sampling.sampling_template import Sampling
from equadratures.sampling.tensorgrid import Tensorgrid
from equadratures.basis import Basis, sparse_grid_basis
import numpy as np
class Sparsegrid(Sampling):
    """
    The class defines a Tensorgrid sampling object.

    :param list parameters: A list of parameters, where each element of the list is an instance of the Parameter class.
    :param Basis basis: An instance of the Basis class corresponding to the multi-index set used.
    """
    def __init__(self, parameters=None, basis=None, orders=None):
        super(Sampling, self).__init__(parameters, basis)
        p, w = self.__get_sparsegrid_quadrature_rule()
        super(Sampling, self).____set_points_and_weights__(p, w)
    def __get_sparse_grid_quadrature_rule(self, orders=None):
        """
        Generates a sparse grid quadrature rule based on the parameters in Poly.

        :param Poly self:
            An instance of the Poly class.
        :param list orders:
            A list of the highest polynomial orders along each dimension.
        :return:
            **x**: A numpy.ndarray of sampled quadrature points with shape (number_of_samples, dimension).

            **w**: A numpy.ndarray of the corresponding quadrature weights with shape (number_of_samples, 1).
        """
        sparse_indices, sparse_factors, not_used = sparse_grid_basis(self.basis.level, self.basis.growth_rule, self.dimensions)
        rows = len(sparse_indices)
        cols = len(sparse_indices[0])

        # For storage we use dictionaries
        individual_tensor_indices = {}
        points_store = {}
        weights_store = {}
        indices = np.zeros((rows))

        for i in range(0,rows):
            orders = sparse_indices[i,:]
            #K, I, points , weights = getPseudospectralCoefficients(self, function, orders.astype(int))
            pts, wts =  self.__get_tensorial_quadrature_rule(orders.astype(int))
            tensor_basis = Basis('Tensor grid', orders.astype(int))
            tensor_elements = tensor_basis.elements
            individual_tensor_indices[i] = tensor_elements
            points_store[i] = pts
            weights_store[i] = wts
            indices[i] = len(I)
        sum_indices = int(np.sum(indices))
        points_saved = np.zeros((sum_indices, dimensions))
        weights_saved = np.zeros((sum_indices))
        counter = int(0)
        for i in range(0,rows):
            for j in range(0, int(indices[i])):
                for d in range(0, dimensions):
                    points_saved[counter,d] = points_store[i][j, d]
                weights_saved[counter] = weights_store[i][j]
                counter = counter + 1
        return points_saved, weights_saved
