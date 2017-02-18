"Regression Example"
from effective_quadratures.polynomial import PolyFit
import numpy as np

# Consider some data!
x_train = np.mat([0,0.0714,0.1429,0.2857,0.3571,0.4286,0.5714,0.6429,0.7143,0.7857,0.9286,1.0000], dtype='float64')
y_train = np.mat([6.8053,-1.5184,1.6416,16.3543,14.3442,16.4426,18.1953,28.9913,27.2246,40.3759,55.3726,72.0], dtype='float64')
x_train = x_train.T
y_train = y_train.T


# Regression using a line!
poly1 = PolyFit(x_train, y_train, 'linear')
x_test = np.arange(0.0, 1.0, 0.01)
poly1.plot(x_test)

# Regression with a quadratic!
poly2 = PolyFit(x_train, y_train, 'quadratic')
poly2.plot(x_test)