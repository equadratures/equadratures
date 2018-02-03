# Sample test utility!
from equadratures import *
import numpy as np


X = np.loadtxt('AirfoilX.data')
fX = np.loadtxt('AirfoilY.data')
m, n = X.shape
fX = np.reshape(fX, (m,1))
parameters = []
totalorders = []
dimensions = n
maxorder = 2
for i in range(0, n):
    parameter = Parameter(param_type='Uniform', lower=-1., upper=1., order=maxorder)
    parameters.append(parameter)
    totalorders.append(maxorder)
basis = Basis('Total order', totalorders)


Poly = Polyreg(training_x=X, training_y=fX, parameters=parameters, basis=basis)

# Eigenvalue plot!
e, W = Poly.computeActiveSubspaces(samples=X)
semilogy_lineplot(np.arange(0, 25), np.abs(e), 'Parameters', 'Eigenvalues')


# Sufficient summary plot!
active1 = np.dot( X , W[:,0:1] )
scatterplot(active1, fX, x_label='w1', y_label='Efficiency')
