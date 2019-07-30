"""Tensor grid based sampling."""
from equadratures.sampling.sampling_template import Sampling
from equadratures.sampling.tensorgrid import Tensorgrid
from equadratures.basis import sparse_grid_basis
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
        # INPUTS
        sparse_indices, sparse_factors, not_used = sparse_grid_basis(self.basis.level, self.basis.growth_rule, self.dimensions)
        rows = len(sparse_indices)
        cols = len(sparse_indices[0])

        # For storage we use dictionaries
        individual_tensor_coefficients = {}
        individual_tensor_indices = {}
        points_store = {}
        weights_store = {}
        indices = np.zeros((rows))

        for i in range(0,rows):
            orders = sparse_indices[i,:]
            K, I, points , weights = getPseudospectralCoefficients(self, function, orders.astype(int))
            individual_tensor_indices[i] = I
            individual_tensor_coefficients[i] =  K
            points_store[i] = points
            weights_store[i] = weights
            indices[i] = len(I)

        sum_indices = int(np.sum(indices))
        store = np.zeros((sum_indices, dimensions+1))
        points_saved = np.zeros((sum_indices, dimensions))
        weights_saved = np.zeros((sum_indices))

        sum_indices = int(np.sum(indices))
        store = np.zeros((sum_indices, dimensions+1))
        points_saved = np.zeros((sum_indices, dimensions))
        weights_saved = np.zeros((sum_indices))
        counter = int(0)
        for i in range(0,rows):
            for j in range(0, int(indices[i])):
                store[counter,0] = sparse_factors[i] * individual_tensor_coefficients[i][j]
                for d in range(0, dimensions):
                    store[counter,d+1] = int(individual_tensor_indices[i][j, d])
                    points_saved[counter,d] = points_store[i][j, d]
                weights_saved[counter] = weights_store[i][j]
                counter = counter + 1
        # Now we use a while loop to iteratively delete the repeated elements while summing up the
        # coefficients!
        index_to_pick = 0
        flag = 1
        counter = 0

        rows = len(store)

        final_store = np.zeros((sum_indices, dimensions + 1))
        while(flag != 0):

            # find the repeated indices
            rep = find_repeated_elements(index_to_pick, store)
            coefficient_value = 0.0

            # Sum up all the coefficient values
            for i in range(0, len(rep)):
                actual_index = rep[i]
                coefficient_value = coefficient_value + store[actual_index,0]

            # Store into a new array
            final_store[counter,0] = coefficient_value
            final_store[counter,1::] = store[index_to_pick, 1::]
            counter = counter + 1

            # Delete index from store
            store = np.delete(store, rep, axis=0)

            # How many entries remain in store?
            rows = len(store)
            if rows == 0:
                flag = 0

        indices_to_delete = np.arange(counter, sum_indices, 1)
        final_store = np.delete(final_store, indices_to_delete, axis=0)

        # Now split final store into coefficients and their index sets!
        coefficients = np.zeros((1, len(final_store)))
        for i in range(0, len(final_store)):
            coefficients[0,i] = final_store[i,0]

        # Splitting final_store to get the indices!
        indices = final_store[:,1::]

        # Now just double check to make sure they are all integers
        for i in range(0, len(indices)):
            for j in range(0, dimensions):
                indices[i,j] = int(indices[i,j])

        K = np.column_stack(coefficients)
        return K, indices, points_saved, weights_saved
