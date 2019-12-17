from equadratures import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from mpl_toolkits.mplot3d import axes3d
VALUE = 15
plt.rcParams.update({'font.size': VALUE})


X = np.loadtxt('../data/design_parameters.dat')
y = np.loadtxt('../data/non_dimensionalized_efficiency.dat')
title = 'Normalised efficiency'

s = Parameter(distribution='uniform', lower=-1., upper=1., order=2)
myparameters = [s for _ in range(0, 25)]
mybasis = Basis('total-order')
mypoly = Poly(parameters=myparameters, basis=mybasis, method='least-squares', \
              sampling_args= {'mesh': 'user-defined', 'sample-points': X, 'sample-outputs': y})
mypoly.set_model()

mysubspace = Subspaces(full_space_poly=mypoly, method='active-subspace')
W = mysubspace.get_subspace()
e = mysubspace.get_eigenvalues()

fig = plt.figure(figsize=(7,4))
ax = fig.add_subplot(111)
plt.semilogy(e, 'o')
plt.ylabel('Eigenvalues (log-scale)')
plt.xlabel('Design parameters')
plt.savefig('../Figures/tutorial_11_fig_a.png', dpi=200, bbox_inches='tight')

true_dimensions = 1
u = X @ W[:, 0:true_dimensions]
fig = plt.figure(figsize=(7,4))
ax = fig.add_subplot(111)
plt.plot(u[:,0], y, 'o', color='gold', markeredgecolor='k', lw=1, ms=13, alpha=0.8)
plt.ylabel(title)
plt.xlabel('u')
plt.savefig('../Figures/tutorial_11_fig_b.png', dpi=200, bbox_inches='tight')

true_dimensions = 2
u = X @ W[:, 0:true_dimensions]
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(u[:,0],u[:,1], y, s=50, c=y, marker='o', edgecolor='k', lw=1, alpha=0.8)
ax.set_xlabel('u1')
ax.set_ylabel('u2')
ax.set_zlabel(title)
plt.savefig('../Figures/tutorial_11_fig_c.png', dpi=200, bbox_inches='tight')


z = X @ W[:,0:true_dimensions] # Projecting on 2D space only!
pts = mysubspace.get_zonotope_vertices()
hull = ConvexHull(pts)

# Plot
fig = plt.figure(figsize=(11,8))
ax = fig.add_subplot(111)
plt.plot(z[:,0], z[:,1], 'o', color='gold', markeredgecolor='k', lw=1, ms=13, alpha=0.8)
for simplex in hull.simplices:
    plt.plot(pts[simplex, 0], pts[simplex, 1], 'k-', lw=2)
plt.savefig('../Figures/tutorial_11_fig_d.png', dpi=200, bbox_inches='tight')