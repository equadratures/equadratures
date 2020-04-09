from equadratures import *
import numpy as np
import matplotlib.pyplot as plt
# Data from before!

dimensions = 1
M = 12
N = 100
p_order = 5
param = Parameter(distribution='Uniform', lower=-1, upper=1., order=p_order)
myParameters = [param for i in range(dimensions)] # one-line for loop for parameters
x_test = np.reshape(np.linspace(-1., 1., N), (N, 1) )
x_train = np.array([0,0.0714,0.1429,0.2857,0.3571,0.4286,0.5714,0.6429,0.7143,0.7857,0.9286,1.0000] )*2. - 1
y_train = np.array([6.8053,-1.5184,1.6416,6.3543,14.3442,16.4426,18.1953,28.9913,27.2246,40.3759,55.3726,72.0] )
x_train = np.reshape(x_train, (M, 1))
y_train = np.reshape(y_train, (M, 1))
noise = 1.5

myBasis = Basis('Univariate')
poly = Poly(myParameters, myBasis)
P = poly.get_poly(x_test).T

# Define the prior!
mu_0 = np.zeros((p_order+1))
Sigma_0 = 100. * np.eye(p_order+1)
samples = 300
coefficients_dist = np.random.multivariate_normal(mu_0, Sigma_0, (samples,p_order+1))

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for i in range(samples):
    plt.plot(x_test, P @ coefficients_dist[i,:].T, '-', alpha=0.1, lw=0.5, color='grey')
for i in range(0, M):
    plt.errorbar(x_train[i,0], y_train[i,0], yerr=noise * 1.96, fmt='o', ecolor='crimson', capthick=1, capsize=8, color='crimson')
plt.xlabel('$X$', fontsize=13)
plt.ylabel('$Y$', fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.ylim([-20, 80])
plt.savefig('../Figures/tutorial_7_fig_a.png' , dpi=200, bbox_inches='tight', pad_inches=0.1)

# Now what if we just add 1 data point to the mix?
values = np.array([0,2,9, 11])
x_use = x_train[values]
y_use = y_train[values]
Sigma_measurements = noise**2 * np.eye(len(values))
P_data = poly.get_poly(x_use).T
Sigma_x = np.linalg.inv( P_data.T @ np.linalg.inv(Sigma_measurements) @ P_data  + np.linalg.inv(Sigma_0) )

mu_x = Sigma_x @   P_data.T @ np.linalg.inv(Sigma_measurements) @ y_use
coefficients_dist = np.random.multivariate_normal(mu_x.flatten(), Sigma_x, (samples,p_order+1))

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for i in range(samples):
    plt.plot(x_test, P @ coefficients_dist[i,:].T, '-', alpha=0.1, lw=0.5, color='grey')
for i in range(0, M):
    plt.errorbar(x_train[i,0], y_train[i,0], yerr=noise * 1.96, fmt='o', ecolor='crimson', capthick=1, capsize=8, color='crimson')
plt.plot(x_test, P @ mu_x, '-', lw=1.5, color='dodgerblue')
plt.xlabel('$X$', fontsize=13)
plt.ylabel('$Y$', fontsize=13)
plt.ylim([-20, 80])
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.savefig('../Figures/tutorial_7_fig_b.png' , dpi=200, bbox_inches='tight', pad_inches=0.1)
