from equadratures import *
import numpy as np
import scipy as sp


## Poly tests
def fun(x):
    return np.exp(x[0])

p_order = 5

x0 = Parameter(param_type="Uniform", lower=-0, upper=1.0, order = p_order)
x1 = Parameter(param_type="Uniform", lower=-0, upper=1.0, order = p_order)
x2 = Parameter(param_type="Uniform", lower=-0, upper=1.0, order = p_order)
x3 = Parameter(param_type="Uniform", lower=-0, upper=1.0, order = p_order)

parameters = [x0,x1,x2,x3]

orders = [p_order,p_order,p_order,p_order]
polybasis = Basis("Total order", orders)
success = 0.0
for i in range(100):
    x_cs = np.random.uniform(size = (int(polybasis.elements.shape[0]/2),len(orders)))
    x_reg = np.random.uniform(size = (1000,len(orders)))

polycs = Polycs(parameters, polybasis, sampling="dlm", fun=fun)
polyreg = Polyreg(parameters, polybasis, training_inputs=x_reg, fun=fun)

coeffs_cs = polycs.coefficients
coeffs_reg = polyreg.coefficients
print np.linalg.norm(coeffs_cs - coeffs_reg)/np.linalg.norm(coeffs_reg)


