from equadratures import *
import numpy as np
import matplotlib.pyplot as plt
VALUE = 15
plt.rcParams.update({'font.size': VALUE})

def rosenbrock_fun(x):
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

mu = 1
sigma = 2
variance = sigma**2
x1 = Parameter(distribution="Gaussian", shape_parameter_A=mu, shape_parameter_B=variance, order=4)
x2 = Parameter(distribution="Gaussian", shape_parameter_A=mu, shape_parameter_B=variance, order=4)
parameters = [x1, x2]

basis = Basis('tensor-grid')
uqProblem = Poly(parameters, basis, method='numerical-integration')
uqProblem.set_model(rosenbrock_fun)
mean, variance = uqProblem.get_mean_and_variance()
print(mean, variance)

large_number = 1000000
s = sigma * np.random.randn(large_number,2) + mu
#f = np.zeros((large_number,1))
f = evaluate_model(s, rosenbrock_fun)
#for i in range(0, large_number):
#    f[i,0] = rosenbrock_fun([s[i,0], s[i,1]])
#print( np.mean(f), np.var(f) )

# Convergence study
mcmc_samples = np.array([10, 100, 1000, 10000, 100000, 500000, 1000000, 5000000])
poly_orders = np.array([1, 2, 3, 4, 5, 6, 7, 8])

poly_means = np.zeros((8))
poly_vars = np.zeros((8))
mcmc_means = np.zeros((8))
mcmc_vars = np.zeros((8))

for i in range(0, 8):
    mu = 1
    sigma = 2
    variance = sigma**2
    x1 = Parameter(distribution="Gaussian", shape_parameter_A=mu, shape_parameter_B=variance, order=i+1)
    x2 = Parameter(distribution="Gaussian", shape_parameter_A=mu, shape_parameter_B=variance, order=i+1)
    parameters = [x1, x2]
    basis = Basis('tensor-grid')
    uqProblem = Poly(parameters, basis, method='numerical-integration')
    uqProblem.set_model(rosenbrock_fun)
    mean_poly, sigma2_poly = uqProblem.get_mean_and_variance()
    poly_means[i] = mean_poly
    poly_vars[i] = sigma2_poly

    large_number = mcmc_samples[i]
    s = sigma * np.random.randn(large_number,2) + mu
    f = evaluate_model(s, rosenbrock_fun)
    mcmc_means[i] = np.mean(f)
    mcmc_vars[i] = np.var(f)
    del x1, x2, parameters, basis, uqProblem, large_number, f, mean_poly, sigma2_poly

print(mcmc_means, mcmc_vars)
print(poly_means, poly_vars)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.loglog(poly_orders**2, poly_means, 'o-', c='crimson', lw=1, ms=8, label='Polynomial' )
plt.loglog(mcmc_samples, mcmc_means, '<-', c='orange', lw=1, ms=8, label='Monte Carlo' )
plt.xlabel('Number of samples', fontsize=VALUE)
plt.ylabel('Mean', fontsize=VALUE)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.20), ncol=3, fancybox=True, shadow=True)
plt.savefig('../Figures/tutorial_4_fig_a.png', dpi=200, bbox_inches='tight')

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.loglog(poly_orders**2, poly_vars, 'o-', c='crimson', lw=1, ms=8, label='Polynomial' )
plt.loglog(mcmc_samples, mcmc_vars, '<-', c='orange', lw=1, ms=8, label='Monte Carlo' )
plt.xlabel('Number of samples', fontsize=VALUE)
plt.ylabel('Variance', fontsize=VALUE)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.20), ncol=3, fancybox=True, shadow=True)
plt.savefig('../Figures/tutorial_4_fig_b.png', dpi=200, bbox_inches='tight')
