#!/usr/bin/env python
from effective_quadratures.parameter import Parameter
from effective_quadratures.indexset import IndexSet
from effective_quadratures.polynomial import Polynomial
from effective_quadratures.utils import meshgrid, twoDgrid, evalfunction
from effective_quadratures.computestats import Statistics
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
def expfun(x):
    return np.exp(x[0] + x[1] ) + 0.5*np.cos(x[0]*2*np.pi)

# Compare actual function with polynomial approximation
s = Parameter(lower=-1, upper=1, points=6)
T = IndexSet('Tensor grid', [5,5])
uq = Polynomial([s,s], T)
        
num_elements = 10
coefficients, index_set, evaled_pts = uq.getPolynomialCoefficients(expfun)
pts, x1, x2 = meshgrid(-1.0, 1.0, num_elements,num_elements)
Approx = uq.getPolynomialApproximation(expfun, pts, coefficients)
A = np.reshape(Approx, (num_elements,num_elements))
gridpts, wts = uq.getPointsAndWeights()
fun = evalfunction(gridpts, expfun)

# Now plot this surface
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(x1, x2, A, rstride=1, cstride=1, cmap=cm.winter,
                       linewidth=0, antialiased=False,  alpha=0.5)
ax.scatter(gridpts[:,0], gridpts[:,1], fun, 'ko')
ax.set_zlim(0, 10)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('Response')

fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

# Now that I have this approximation, how do I get statistics?
