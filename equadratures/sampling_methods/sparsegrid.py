"""Sparse grid based sampling."""
from equadratures.sampling_methods.sampling_template import Sampling
from equadratures.sampling_methods.tensorgrid import Tensorgrid
from equadratures.basis import Basis
import numpy as np
class Sparsegrid(Sampling):
    """
    The class defines a Tensorgrid sampling object.

    :param list parameters: A list of parameters, where each element of the list is an instance of the Parameter class.
    :param Basis basis: An instance of the Basis class corresponding to the multi-index set used.
    """
    def __init__(self, parameters, basis):
        self.parameters = parameters
        self.basis = basis
        self.dimensions = len(self.parameters)
        self.__set_sparsegrid_quadrature_rule()
        super(Sparsegrid, self).__init__(self.parameters, self.basis, self.points, self.weights)
    def __set_sparsegrid_quadrature_rule(self, orders=None):
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
        sparse_indices, sparse_factors, not_used = self.basis.get_basis()
        rows = len(sparse_indices)
        cols = len(sparse_indices[0])

        # For storage we use dictionaries
        points_store = {}
        weights_store = {}
        indices = np.zeros((rows))
        self.tensor_product_list = []
        for i in range(0,rows):
            orders = sparse_indices[i,:]
            myBasis = Basis('tensor-grid')
            myTensor = Tensorgrid(parameters=self.parameters, basis=myBasis, orders=orders.astype(int) )
            self.tensor_product_list.append(myTensor)
            pts = myTensor.points
            wts = myTensor.weights * sparse_factors[i]
            points_store[i] = pts
            weights_store[i] = wts
            indices[i] = myTensor.basis.cardinality
            del myTensor, myBasis
        sum_indices = int(np.sum(indices))
        points_saved = np.zeros((sum_indices, self.basis.dimensions))
        weights_saved = np.zeros((sum_indices))
        counter = int(0)
        for i in range(0,rows):
            for j in range(0, int(indices[i])):
                for d in range(0, self.basis.dimensions):
                    points_saved[counter,d] = points_store[i][j, d]
                weights_saved[counter] = weights_store[i][j]
                counter = counter + 1
        self.points , indices = np.unique(points_saved, axis=0, return_index=True)
        self.weights = weights_saved[indices]
        self.sparse_indices = sparse_indices
        self.sparse_weights = sparse_factors