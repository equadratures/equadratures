from equadratures import *
import numpy as np
import matplotlib.pyplot as plt
VALUE = 15
plt.rcParams.update({'font.size': VALUE})

order = 4
s1 = Parameter(lower=-1, upper=1, order=order, distribution='Uniform')
myBasis = Basis('univariate')
myPoly = Poly(s1, myBasis, method='numerical-integration')
points, weights = myPoly.get_points_and_weights()
def function(x):
    return x[0]**7 - 3.0*x[0]**6 + x[0]**5 - 10.0*x[0]**4 +4.0
    
integral = float( 2  * np.dot(weights , evaluate_model(points, function) ) )
print(integral)

s2 = Parameter(lower=-1, upper=1, order=order, distribution='uniform', endpoints='lower')
s3 = Parameter(lower=-1, upper=1, order=order, distribution='uniform', endpoints='upper')
s4 = Parameter(lower=-1, upper=1, order=order, distribution='uniform', endpoints='both')

myPoly2 = Poly(s2, myBasis, method='numerical-integration')
myPoly3 = Poly(s3, myBasis, method='numerical-integration')
myPoly4 = Poly(s4, myBasis, method='numerical-integration')

zeros = np.zeros((order+1))
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.xlabel('$s$', fontsize=VALUE)
plt.ylabel('Quadrature points', fontsize=VALUE)
plt.plot(myPoly.get_points(), zeros, 'o', c='crimson', lw=4, ms=15)
plt.plot(myPoly2.get_points(), zeros-0.1, '<', c='orange', lw=4, ms=15)
plt.plot(myPoly3.get_points(), zeros+0.1, '>', c='navy', lw=4, ms=15)
plt.plot(myPoly4.get_points(), zeros+0.2, 's', c='limegreen', lw=4, ms=15)
plt.grid()
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
plt.savefig('../Figures/tutorial_2_fig_a.png', dpi=200, bbox_inches='tight')

