from equadratures import *
import numpy as np


def fun(x):
    return np.exp(x[0] + x[1])

p_order = 15
x0 = Parameter(param_type="Uniform", lower=-1., upper=1.0, order = p_order)
x1 = Parameter(param_type="Uniform", lower=-1., upper=1.0, order = p_order)
parameters = [x0,x1]
polybasis = Basis("Total order")
for i in range(100):
    x_reg = np.random.uniform(size = (5000,2))

polyreg = Polyreg(parameters, polybasis, training_inputs=x_reg, fun=fun)
coeffplot2D(polyreg.coefficients, polybasis.elements, '$i_1$', '$i_2$')