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

mysubspace = Subspaces(method='variable-projection', sample_points=X, sample_outputs=y)
W = mysubspace.get_subspace()
true_dimensions = 2
u = X @ W[:, 0:true_dimensions]

# Plot
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(u[:,0],u[:,1], y, s=50, c=y, marker='o', edgecolor='k', lw=1, alpha=0.8)
ax.set_xlabel('u1')
ax.set_ylabel('u2')
ax.set_zlabel(title)
plt.savefig('../Figures/tutorial_12_fig_a.png', dpi=200, bbox_inches='tight')