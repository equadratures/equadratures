from unittest import TestCase
import unittest
from equadratures import *
import numpy as np


class TestPolyreg(TestCase):

    def test_simple2D(self):
        d = 5
        param = Parameter(distribution='Uniform', lower=-1, upper=1., order=1)
        myParameters = [param for _ in range(d)]
        def f(x):
            return x[0] * x[1]

        x_train = np.array([[7.58632788e-01, 4.81746227e-01, 5.02577142e-01,
                7.67376530e-01, 4.90829684e-01],
               [1.98916966e-01, 8.53442257e-01, 4.65585866e-01,
                2.75222052e-01, 6.77784764e-01],
               [7.46828043e-01, 8.58487468e-01, 4.32075141e-01,
                1.42985459e-01, 6.25679567e-01],
               [7.35825598e-01, 1.65463815e-01, 9.13499589e-01,
                4.86974147e-04, 1.38084505e-01],
               [1.66053494e-01, 8.26502987e-01, 9.81150618e-01,
                4.65587483e-01, 5.69055172e-01],
               [8.41720170e-01, 4.21238174e-01, 7.42375218e-01,
                8.41220207e-02, 2.07048763e-01],
               [5.80581970e-01, 4.52048112e-01, 3.92967568e-01,
                7.83143576e-01, 7.76403603e-01],
               [9.74079876e-01, 8.72576146e-01, 2.10026353e-01,
                4.08982657e-01, 1.89006589e-01],
               [4.44494044e-01, 5.58853652e-01, 2.25635327e-01,
                3.94315874e-01, 1.49055844e-01],
               [2.67176489e-01, 7.36300543e-01, 9.07632137e-01,
                5.03907567e-01, 3.31995486e-01],
               [7.89158773e-01, 6.31673466e-01, 5.23065889e-01,
                8.48395576e-02, 6.66838037e-01],
               [8.71387227e-01, 3.02452797e-02, 3.66761253e-01,
                2.98375233e-02, 8.16636350e-01],
               [4.09188935e-01, 7.23745682e-01, 2.70466646e-01,
                3.33145142e-01, 1.17563309e-01],
               [2.86957871e-01, 9.83273435e-01, 9.50085865e-01,
                4.25726126e-01, 7.05275218e-01],
               [1.56317650e-01, 1.73866379e-01, 7.74967016e-01,
                6.37677812e-01, 7.72158379e-01]])
        polynomialOrders = np.full(d, 2)
        myBasis = Basis('Total order', polynomialOrders)
        poly = Polycs(myParameters, myBasis, training_inputs=x_train, fun=f)
        actual_coeffs = np.zeros(myBasis.cardinality)
        actual_coeffs[-2] = 1.0/3.0

        np.testing.assert_almost_equal(np.linalg.norm(actual_coeffs - poly.coefficients.flatten()), 0, decimal=4,
                                       err_msg="Difference greated than imposed tolerance for coeffs")


if __name__ == '__main__':
    unittest.main()
