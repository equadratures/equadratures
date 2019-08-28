from equadratures import *
import numpy as np
import matplotlib.pyplot as plt
VALUE = 15
plt.rcParams.update({'font.size': VALUE})

order = 5
s1 = Parameter(lower=-1, upper=1, order=order, distribution='Uniform')
myBasis = Basis('univariate')
myPoly = Poly(s1, myBasis, method='numerical-integration')
xo = np.linspace(-1., 1, 100)
P = myPoly.get_poly(xo)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.plot(xo, P[0,:], lw=2, label='Order 0')
plt.plot(xo, P[1,:], lw=2, label='Order 1')
plt.plot(xo, P[2,:], lw=2, label='Order 2')
plt.plot(xo, P[3,:], lw=2, label='Order 3')
plt.plot(xo, P[4,:], lw=2, label='Order 4')
plt.plot(xo, P[5,:], lw=2, label='Order 5')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.30), ncol=3, fancybox=True, shadow=True)
plt.xlabel('$s$', fontsize=18)
plt.ylabel('Polynomials', fontsize=18)
plt.savefig('../Figures/tutorial_3_fig_a.png', dpi=200, bbox_inches='tight')

factor_0 = 1.
factor_1 = 1.0 / np.sqrt(2.0 * 1.0 + 1.)
factor_2 = 1.0 / np.sqrt(2.0 * 2.0 + 1.)
factor_3 = 1.0 / np.sqrt(2.0 * 3.0 + 1.)
factor_4 = 1.0 / np.sqrt(2.0 * 4.0 + 1.)
factor_5 = 1.0 / np.sqrt(2.0 * 5.0 + 1.)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.plot(xo, factor_0 * P[0,:], lw=2, label='Order 0')
plt.plot(xo, factor_1 * P[1,:], lw=2, label='Order 1')
plt.plot(xo, factor_2 * P[2,:], lw=2, label='Order 2')
plt.plot(xo, factor_3 * P[3,:], lw=2, label='Order 3')
plt.plot(xo, factor_4 * P[4,:], lw=2, label='Order 4')
plt.plot(xo, factor_5 * P[5,:], lw=2, label='Order 5')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.30), ncol=3, fancybox=True, shadow=True)
plt.xlabel('$s$', fontsize=18)
plt.ylabel('Scaled polynomials', fontsize=18)
plt.savefig('../Figures/tutorial_3_fig_b.png', dpi=200, bbox_inches='tight')
