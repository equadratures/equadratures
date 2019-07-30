"""The mesh class."""
from equadratures.parameter import Parameter
from equadratures.basis import Basis
from equadratures.subsampling import qr_column_pivoting, svd_subset_selection, newton_determinant_maximization
from equadratures.sparse_and_tensor_quadrature import sparse_grid, tensor_grid
import numpy as np
class Samples(object):
    """
    Generates a mesh.
    """
    def __init__(self, polynomial):
        mesh = polynomial.samples[0]
        self.polynomial = polynomial

        # Setup the cases!
        if mesh.lower() == 'tensor-grid' and polynomial.samples[1].get('subsampling-option').lower() == 'random':
            case = 'A'
        if mesh.lower() == 'tensor-grid' and polynomial.samples[1].get('subsampling-option').lower() ~= 'random':
            case = 'B'
        elif mesh.lower() == 'sparse-grid':
            case = 'C'
        elif mesh.lower() == 'monte-carlo':
            case = 'D'
        elif mesh.lower() == 'christoffel':
            case = 'E'
        elif mesh.lower() == 'latin-hyper-cube'
            case = 'F'
        else:
            raise(ValueError, 'Unrecognized string argument for samples tuple.')

        if not polynomial.samples[1]:
            self.__quadrature_points = pts
            self.__quadrature_weights = wts
        else:
            subsampling_ratio = float(polynomial.samples[1].get('subsampling-ratio'))
            n =
            if polynomial.samples[1].get('subsampling-option').lower() == 'qr':
                subsampled_pts, subsampled_wts = self.qr_column_pivoting(pts, wts)
            elif polynomial.samples[1].get('subsampling-option').lower() == 'svd':
                subsampled_pts, subsampled_wts = self.svd_subset_selection(pts, wts)
            elif polynomial.samples[1].get('subsampling-option').lower() == 'newton':
                subsampled_pts, subsampled_wts = self.convex_relaxation(pts, wts)
            else:
                raise(ValueError, 'Unrecognized dict argument for subsampling-option.')
            self.__quadrature_points = subsampled_pts
            self.__quadrature_weights = subsampled_wts
    def get_points_and_weights(self):
        """
        Returns the points and weights associated with the

        """
        return self.__quadrature_points, self.__quadrature_weights
    def get_monte_carlo_quadrature(self):
        pts = np.zeros((m_big, self.polynomial.dimensions))
        for i in range(0, self.dimensions):
            univariate_samples = self.polynomial.parameters[i].get_samples(m_big)
            for j in range(0, m_big):
                pts[j, i] = univariate_samples[j]
        wts =  1.0/(np.sum( super(Poly, self).get_poly(pts)**2 , 0) )**2
        wts = wts * 1.0/np.sum(wts)
        return pts, wts
