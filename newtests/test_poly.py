from equadratures import *
import numpy as np


p1 = Parameter(param_type='Gaussian', shape_parameter_A=0., shape_parameter_B=0.5, order=4)
pts = np.linspace(-2., 2., 50)
poly, poly_grad = p1._getOrthoPoly(pts)
polynomialplot(poly, pts)

p, w = p1._getLocalQuadrature()

print p, w
print np.sum(w)