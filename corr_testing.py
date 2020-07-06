from equadratures import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, norm
from scipy.linalg import lu
from corr_new_stuff import get_R, get_Phi, get_coeffs, poly_eval, l2_sample_w

#
def genz(z):
    e = 0
    c = 3.5 * 2 * np.pi * np.ones(2)
    return np.cos(2 * np.pi * e + np.dot(c, z))
#%%
d = 2
parameters = [Parameter(distribution='beta', shape_parameter_A=5.0, shape_parameter_B=2.0,
               lower=0.0, upper=1.0, order=29) for _ in range(d)]
cov_mat = np.array([ [ 1.0,  -0.855],
               [ -0.855,  1.0]])
basis = Basis('tensor-grid')
uncorr_poly = Poly(parameters, basis)

corr = Correlations(uncorr_poly, cov_mat)
# print(corr.R0)
#
R = get_R(corr, S=5000)
# print(np.linalg.cond(R))

Phi, S_samples = get_Phi(corr, R, S=5000)
coeffs, Z, Phi2, V = get_coeffs(Phi, S_samples, genz)

approx = lambda x:poly_eval(x, corr, R, coeffs)
w = lambda x:corr.get_pdf(x)

N = 1000
# err, fxx, gxx, wxx, xx = l2(N, d, w, approx, genz, 0.4, 0.99, plot=True)
err = l2_sample_w(N, corr, approx, genz)
print(err)

# print(err)
# fig, ax = plt.subplots()
# ax.scatter(xx[:,0], xx[:,1], c=wxx)
# ax.scatter(xx[:,0], xx[:,1], c=fxx)
# ax.scatter(xx[:,0], xx[:,1], c=gxx)
# ax.scatter(xx[:,0], xx[:,1], c=wxx)
# ax.plot(xx, fxx)
# ax.plot(xx, gxx)
# ax.plot(xx, wxx)
# ax.scatter(Z[:,0], Z[:,1], c=approx(Z), s=50)
# ax.set_ylim([-1,3])
# plt.show()

#%% compare with uncorr poly using uncorrelated quadrature?
uncorr_poly.set_model(genz)
err_uncorr = l2_sample_w(N, corr, uncorr_poly.get_polyfit_function(), genz)
print(err_uncorr)

#%%
Z = corr.get_correlated_samples(N=)
uncorr_poly = Poly(parameters, basis, method='least-squares', )

#%% uncorr with leja? It probably won't be better... because quadrature is optimal

#%% compare with nataf?
d = 2
S = 5000

G_nataf = np.random.multivariate_normal(np.zeros(d), np.eye(d), size=S)
def f_Tinv(func, corr, G):
    # assume G is normally distributed
    if len(G.shape) == 1:
        G = G.reshape(1,-1)
    Z_nataf = corr.get_correlated_from_uncorrelated(G)
    return np.apply_along_axis(func, 1, Z_nataf)
parameters_gaussian = [Parameter(distribution='gaussian', shape_parameter_A=0.0,
                                 shape_parameter_B=1.0, order=15) for _ in range(d)]
poly_gaussian = Poly(parameters_gaussian, basis, method='numerical-integration')

# Phi_nataf = poly_gaussian.get_poly(G_nataf).T
# coeffs_nataf = get_coeffs(Phi_nataf, G_nataf, lambda g:f_Tinv(genz, corr, g))[0]
# poly_gaussian._set_coefficients(coeffs_nataf)
poly_gaussian.set_model(lambda g:f_Tinv(genz, corr, g))


def gT(func, corr, Z):
    # Z ~ omega
    if len(Z.shape) == 1:
        Z = Z.reshape(1,-1)
    d = Z.shape[1]

    poly = corr.poly
    U_corr = np.zeros(Z.shape)
    for i in range(d):
        U_corr[:, i] = poly.parameters[i].get_cdf(Z[:, i])
    G_corr = np.zeros(U_corr.shape)
    for i in range(d):
        G_corr[:, i] = norm.ppf(U_corr[:, i])
    G_uncorr = G_corr @ np.linalg.inv(corr.A.T)
    # print(func(G_uncorr).reshape(-1))
    # return np.apply_along_axis(func, 1, G_uncorr)
    return func(G_uncorr).reshape(-1)

N = 1000
approx = lambda x:gT(poly_gaussian.get_polyfit_function(), corr, x)
err_nataf = l2_sample_w(N, corr, approx, genz)
print(err_nataf)

#%% uncorr density
# test_points = np.linspace(0.01, 0.99, 100)
# xx, yy = np.meshgrid(test_points, test_points)
# zz = parameters[0].get_pdf(xx) * parameters[1].get_pdf(yy)
# fig, ax = plt.subplots()
# ax.scatter(xx, yy, c=zz)
# ax.scatter(Z[:,0], Z[:,1])
# plt.show()

#%%
# # A model function
# def function(s):
#     s1 = s[0]
#     s2 = s[1]
#     return s1**2 - s2**3 - 2. * s1 - 0.5 * s2
#
# # The standard setup.
# # param_s1 = Parameter(distribution='Gaussian', shape_parameter_A=0., shape_parameter_B=1., order=3)
# # param_s2 = Parameter(distribution='Gaussian', shape_parameter_A=0., shape_parameter_B=1., order=3)
# param_s1 = Parameter(distribution='Uniform', lower=0., upper=1., order=3)
# param_s2 = Parameter(distribution='Uniform', lower=0., upper=1., order=3)
# # param_s1 = Parameter(distribution='Gaussian', shape_parameter_A=0., shape_parameter_B=1., order=3)
# # param_s2 = Parameter(distribution='Exponential', shape_parameter_A=1., order=3)
# cov_mat = np.array([ [ 1.0,  0.5],
#                [ 0.5,  1.0]])
# # basis = Basis('sparse-grid', level=3, growth_rule='exponential')
# basis = Basis('tensor-grid')
# uncorr_poly = Poly([param_s1, param_s2], basis, method='numerical-integration')
#
#
# def g(z):
#     e = 0
#     c = 3.5 * 2 * np.pi
#     return np.cos(2 * np.pi * e + np.dot(c, z))
#
# s = [Parameter(distribution='beta', shape_parameter_A=10.0, shape_parameter_B=10.0,
#                lower=0.0, upper=1.0, order=7)]
# cov_mat = 1.0
# basis = Basis('univariate')
# uncorr_poly = Poly(s, basis, method='numerical-integration')
#
#
# corr = Correlations(uncorr_poly, cov_mat)
# R = get_R(corr)
# Phi, S = get_Phi(corr, R)
# coeffs, Z, Phi2, V = get_coeffs(Phi, S, g)

# print(poly_eval(Z, corr, R, coeffs))
# print(np.apply_along_axis(g, 1, Z).reshape(-1))

# approx = lambda x:poly_eval(x, corr, R, coeffs)
# w = lambda x:omega(x, corr)
# N = 100
# d = 1
# err = l2(N, d, w, approx, g, 0.01, 0.99)

#%%
# err, fxx, gxx, wxx, xx = l2(N, d, w, approx, g, 0.01, 0.99, plot=True)
# print(err)
# fig, ax = plt.subplots()
# ax.plot(xx, fxx)
# ax.plot(xx, gxx)
# ax.plot(xx, wxx)
# ax.scatter(Z, approx(Z))
# ax.set_ylim([-1,3])
# plt.show()

#%% Compare with gauss quadrature for uniform
# unif_param = [Parameter(distribution='uniform', lower=0.0, upper=1.0, order=7)]
# unif_poly = Poly(unif_param, basis, method='numerical-integration')
# Z_unif = unif_poly.get_points()
#
# Phi_unif = unif_poly.get_poly(Z_unif).T
# V_unif = 1.0/np.linalg.norm(Phi_unif, axis=1)
# y_unif = np.apply_along_axis(g, 1, Z_unif).reshape(-1)
# coeffs_unif = np.linalg.solve(np.diag(V_unif) @ Phi_unif, V_unif*y_unif)
# unif_poly.coefficients = coeffs_unif.copy()
#
# approx_unif = unif_poly.get_polyfit_function()
# err_unif, fxx_unif, _, _, _ = l2(N, d, w, approx_unif, g, 0.01, 0.99, plot=True)
# fig, ax = plt.subplots()
# ax.plot(xx, fxx_unif)
# ax.plot(xx, gxx)
# ax.plot(xx, wxx)
# ax.scatter(Z_unif, approx_unif(Z_unif))
# # ax.set_ylim([-1,3])
# plt.show()

#%%





# Now feed this polynomial to the Correlations class along with the correlation matrix.
# corr = Correlations(poly, R)
# corr.set_model(function)
# poly2 = corr.get_transformed_poly()
# #%%
# uncorr_poly.set_model(function)
# uncorr_approx = uncorr_poly.get_polyfit_function()
# w = lambda x:omega(x, corr)
# N = 100
# d = 2
# err = l2(N, d, w, approx, function, 0.01, 0.99)
# print(err)
#
# #%%
# N_test = 100
# Z_test = np.zeros((N_test**2, 2))
# Z_test[:,0] = np.repeat(np.linspace(-4,4, N_test), N_test)
# Z_test[:,1] = np.tile(np.linspace(0,2, N_test), N_test)
#
#
# #%%
# print(omega(Z_test, poly))
# plt.figure()
# plt.scatter(Z_test[:,0], Z_test[:,1], c=omega(Z_test, poly))
# plt.show()
#
# #%%
# Z = np.random.multivariate_normal(np.zeros(2), np.eye(2), size=10000)
# Z_corr = corr.get_correlated_from_uncorrelated(Z)
# print(np.corrcoef(Z_corr.T))