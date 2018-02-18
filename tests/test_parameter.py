# Sample test utility!
from equadratures import *
import numpy as np

# Experiment 1
p1 = Parameter(param_type='Uniform', lower=-1.0, upper=5.5, order=3)
p, w = p1._getLocalQuadrature()
P , grad_P = p1._getOrthoPoly(p)
print P

p2 = Parameter(param_type='Uniform', lower=-1.0, upper=1.0, order=3)
po, wo = p2._getLocalQuadrature()
P2 , grad_P2 = p2._getOrthoPoly(po)
print P2

print 'Quadrature points'
print p
print po

x, y = p1.getPDF(50000)
yy = p1.getSamples()
histogram(yy, 'Samples', 'Histogram')

x, y = p2.getPDF(50000)
yy = p2.getSamples()
histogram(yy, 'Samples', 'Histogram')
del p, po, P, P2

# Experiment 2
p1 = Parameter(param_type='Beta', lower=-1.0, upper=5.5, shape_parameter_A = 3.0, shape_parameter_B = 2.0, order=3)
p, w = p1._getLocalQuadrature()
P , grad_P = p1._getOrthoPoly(p)
print P

p2 = Parameter(param_type='Beta', lower=0.0, upper=1.0, shape_parameter_A = 3.0, shape_parameter_B = 2.0, order=3)
po, w = p2._getLocalQuadrature()
P2 , grad_P2 = p2._getOrthoPoly(po)
print P2

print 'Quadrature points'
print p
print po

x, y = p1.getPDF(50000)
lineplot(x, y, 'Samples', 'Kernel density estimate')

x, y = p2.getPDF(50000)
lineplot(x, y, 'Samples', 'Kernel density estimate')