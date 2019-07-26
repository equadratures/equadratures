from unittest import TestCase
import unittest
from equadratures import *
import numpy as np


class TestPolycs(TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(0)

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
    @staticmethod
    def fs(x):
        return np.sum(x)

    def test_input_errors(self):
        dims = 3
        p_order = 2
        myBasis = Basis('tensor grid', [p_order for _ in range(dims)])
        myParams = [Parameter(p_order, distribution='uniform', lower=-1.0, upper=1.0) for _ in range(dims)]
        x = np.random.uniform(low=-1.0, size=(100, 3))
        x4 = np.random.uniform(low=-1.0, size=(100, 4))
        y = np.apply_along_axis(self.fs, 1, x)
        g = 0

        def e1():
            Polycs(myParams, myBasis, training_outputs=y) # no self.x for data driven
        def e2():
            Polycs(myParams, myBasis, fun=self.fs) # no sampling method
        def e3():
            Polycs(myParams, myBasis, training_inputs=x, training_outputs=y, fun=self.fs) # can't have both y and f
        def e4():
            Polycs(myParams, myBasis, training_inputs=x, fun=g) # bad function
        def e5():
            Polycs(myParams, myBasis, training_inputs=x4, fun=self.fs) # wrong dimension
        def e6():
            Polycs(myParams, myBasis, fun=self.fs, sampling='invalid') # bad sampling method

        self.assertRaises(ValueError, e1)
        self.assertRaises(ValueError, e2)
        self.assertRaises(ValueError, e3)
        self.assertRaises(ValueError, e4)
        self.assertRaises(ValueError, e5)
        self.assertRaises(ValueError, e6)

    def test_sampling(self):
        dims = 3
        p_order = 2
        myBasis = Basis('tensor grid', [p_order for _ in range(dims)])
        myParams = [Parameter(p_order, distribution='uniform', lower=-1.0, upper=1.0) for _ in range(dims)]
        myParams_g = [Parameter(p_order, distribution='gaussian', shape_parameter_A=0.0, shape_parameter_B=1.0)
                      for _ in range(dims)]

        poly_std = Polycs(myParams, myBasis, fun=self.fs, sampling="standard", no_of_points=20)
        poly_asm_u = Polycs(myParams, myBasis, fun=self.fs, sampling="asymptotic", no_of_points=20)
        poly_asm_g = Polycs(myParams_g, myBasis, fun=self.fs, sampling="asymptotic", no_of_points=20)
        poly_dlm = Polycs(myParams, myBasis, fun=self.fs, sampling="dlm", no_of_points=20)

        self.assertEqual(poly_std.x.shape, (20, 3))
        self.assertEqual(poly_asm_u.x.shape ,(20, 3))
        self.assertEqual(poly_asm_g.x.shape,(20, 3))
        self.assertEqual(poly_dlm.x.shape ,(20, 3))

        xi_2 = np.linalg.norm(poly_asm_g.x, axis=1) ** 2
        np.testing.assert_array_almost_equal(np.exp(-xi_2 / 4.0), np.diag(poly_asm_g.w))

        np.testing.assert_array_almost_equal(np.prod((1 - poly_asm_u.x ** 2) ** .25, axis=1), np.diag(poly_asm_u.w))

        p, w = poly_dlm.getQuadratureRule(options='tensor grid')
        num_points = 20
        for i in range(num_points):
            match = np.where((poly_dlm.x[i] == p).all(axis=1))[0]
            # assert shape[0] equal to 1
            matched_index = match[0]
            np.testing.assert_almost_equal(np.sqrt(w)[matched_index] , np.diag(poly_dlm.w)[i])

if __name__ == '__main__':
    unittest.main()
