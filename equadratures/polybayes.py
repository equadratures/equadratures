import numpy as np
from scipy.optimize import minimize, basinhopping, Bounds
from copy import deepcopy
from time import time

class Coefficients(object):
    """
    A class for coefficients.
    """
    def __init__(self, prior_mu, prior_cov, sigma_data=1e-7):
        self.prior_mu = prior_mu
        self.prior_cov = prior_cov
        self.inv_prior_cov = np.linalg.inv(self.prior_cov)
        self.sigma_data = sigma_data # nugget / uncertainty in model output!
        self.posterior_mu = prior_mu.copy()
        self.posterior_cov = prior_cov.copy()

    def compute_posterior_coefficients(self, polyobj, x_train, y_train):
        assert x_train.shape[0] == y_train.shape[0]
        P = polyobj.get_poly(x_train)
        self.P = P
        # TODO: weights?
        self.A = 1./self.sigma_data**2 * P @ P.T + self.inv_prior_cov
        self.posterior_cov = np.linalg.inv(1./self.sigma_data**2 * P @ P.T + self.inv_prior_cov )
        self.posterior_mu = self.posterior_cov @ (1./self.sigma_data**2 * P @ y_train + \
                                                self.inv_prior_cov @ self.prior_mu)
        self.x_train = x_train
        self.y_train = y_train

class Polybayes(object):
    """
    A Bayesian wrapper for the Poly class.
    """
    def __init__(self, coefficients, polyobj):
        self.coefficients = coefficients
        self.polyobj = polyobj

    def get_poly_prior(self, x_test):
        """
        Evaluate mean and covariance of function values at x_test using prior coefficient distribution
        :param x_test:
        :return:
        """
        P = self.polyobj.get_poly(x_test)
        mu = self.coefficients.prior_mu
        Sigma = self.coefficients.prior_cov
        polymu = P.T @ mu.reshape(len(mu), 1)
        polycov = P.T @ Sigma @ P
        return polymu, polycov

    def get_poly_posterior(self, x_test, coeff_mu=None, coeff_var=None):
        """
        Evaluate mean and covariance of function values at x_test using posterior coefficient distribution
        :param x_test:
        :return:
        """
        P = self.polyobj.get_poly(x_test)
        if coeff_mu is None:
            mu = self.coefficients.posterior_mu
        if coeff_var is None:
            Sigma = self.coefficients.posterior_cov
        polymu = P.T @ mu.reshape(len(mu), 1)
        polycov = P.T @ Sigma @ P
        return polymu, polycov

    def get_monte_carlo_mean(self):
        """
        Compute E_x[g(x)] using Monte Carlo (on just x not coeffs_0).
        """
        d = self.polyobj.dimensions
        N_mc = 30000
        xx = np.zeros((N_mc, d))
        for dd in range(d):
            xx[:, dd] = self.polyobj.parameters[dd].get_samples(N_mc)
        V_xx = self.polyobj.get_poly(xx).T
        v_mu = np.mean(V_xx, axis=0).reshape(-1, 1)
        mu = self.coefficients.posterior_mu.reshape(-1, 1)
        Sigma = self.coefficients.posterior_cov
        g_mu, g_cov = v_mu.T @ mu, v_mu.T @ Sigma @ v_mu
        return float(g_mu), np.sqrt(float(g_cov))

    def get_poly_chaos_mean(self):
        """
        Returns the polynomial chaos mean wrt x using the posterior coefficients.
        """
        # def polyfunc(x):
        #     P = self.polyobj.get_poly(x)
        #     mu, _ = self.get_poly_posterior(x)
        #     return mu
        #
        # # Create a polynomial instance
        # param = self.polyobj.parameters
        # basis = self.polyobj.basis
        # poly = eq.Poly(param, basis, method=self.polyobj.method)
        # poly.set_model(polyfunc)
        # return poly.get_mean_and_variance()
        mu = self.coefficients.posterior_mu
        Sigma = self.coefficients.posterior_cov
        return mu[0], np.sqrt(Sigma[0, 0])

    def get_new_input_location(self, method='variance', x_train=None, y_train=None, opt_method='CG',
                               disp=False):
        """
        Returns a new location based on an optimisation. Note that the
        training data provided to the Coefficients is used. This can
        be overwritten if required.

        For multi-fidelity useage, typically the high-fidelity data is used.
        """
        if method.lower() == 'variance':
            function = lambda value: posterior_variance_opt(self, value)
        elif method.lower() == 'det':
            function = lambda value: det_of_posterior(self, value)
        else:
            if x_train is None:
                x_train = self.coefficients.x_train
                y_train = self.coefficients.y_train
            function = lambda value: marginal_likelihood(self, value, x_train, y_train)

        x0 = np.zeros(self.polyobj.dimensions)
        for i in range(self.polyobj.dimensions):
            x0[i] = self.polyobj.parameters[i].mean

        class MyBounds(object):
            def __init__(self, xmax, xmin):
                self.xmax = np.array(xmax)
                self.xmin = np.array(xmin)

            def __call__(self, **kwargs):
                x = kwargs["x_new"]
                tmax = bool(np.all(x <= self.xmax))
                tmin = bool(np.all(x >= self.xmin))
                return tmax and tmin

        bounds = [(self.polyobj.parameters[i].lower, self.polyobj.parameters[i].upper)
                  for i in range(self.polyobj.dimensions)]

        mybounds = MyBounds([bounds[i][1] for i in range(self.polyobj.dimensions)]
                            , [bounds[i][0] for i in range(self.polyobj.dimensions)])
        # print(mybounds)
        minimizer_kwargs = {'method': opt_method, 'jac': True, 'bounds': bounds}
        opt_ret = basinhopping(function, x0=x0, minimizer_kwargs=minimizer_kwargs, disp=disp,
                               accept_test=mybounds)

        return opt_ret.x

    def get_integral_poly_density(self):
        """
        \int Sigma_g rho(x)dx
         = \int P(x).T * Sigma_w * P(x') * rho(x) dx
         = \int P(x) * rho(x) * dx * Sigma_w * P(x')
         = numerical_int(P(x) * rho(x) ) * Sigma_w * P(x')
        """
        #quad = Quadrature(parameters=poly.parameters, basis=poly.basis, \
        #                    points=poly.inputs, mesh='monte-carlo', corr=None, oversampling=50.0)
        pt, wt = self.polyobj.get_points_and_weights()

        # Perhaps we should re-visit this...increase order? PS 2/10/20
        self.quad_pts = pt
        self.quad_wts = wt
        self.P = self.polyobj.get_poly(self.quad_pts)
        int_P = np.zeros((self.P.shape[0], 1))
        for i in range(0, self.P.shape[0]):
            int_P[i] = self.P[i,:] @ self.quad_wts
        self.int_P = int_P
        return self.int_P.T @ self.coefficients.posterior_mu

    def get_integral_integral_poly_density(self):
        """
        \int Sigma_g rho(x)dx
         = \int \int  P(x).T * Sigma_w * P(x') * rho(x) * rho(x') dx
         = numerical_int(P(x) * rho(x) ) * Sigma_w * P(x')
        """
        return float( self.int_P.T @ self.coefficients.posterior_cov @ self.int_P )

    def get_poly_posterior_mean(self):
        self.mean_of_mean = float( self.get_integral_poly_density() )
        self.cov_of_mean = float( self.get_integral_integral_poly_density() )
        return self.mean_of_mean, self.cov_of_mean

    def get_poly_posterior_var(self):
        """
        \int rho(x)^2 * mu_g(x) dx - mu^2
        """
        #int_P_squared = np.zeros((self.P.shape[0], 1))
        #for i in range(0, self.P.shape[0]):
        #    int_P_squared[i] = self.P[i,:] @ np.diag(self.quad_wts) @ self.quad_wts

        int_P_squared = np.zeros((self.P.shape[0], 1))
        for i in range(0, self.P.shape[0]):
            int_P_squared[i] = self.P[i,:]**2 @ self.quad_wts

        mean_of_var = float( (int_P_squared.T @ self.coefficients.posterior_mu**2)  - self.mean_of_mean**2 )
        var_of_var = float( int_P_squared.T @ self.coefficients.posterior_cov @ int_P_squared)

        return mean_of_var, var_of_var
        
    def condition_on_mean(self, x_test, mean_val, mean_var=0.0):
        """
        Evaluate mean and (co?)variance function of output conditioned on a spatial mean value
        using posterior coefficient distribution
        :param mean_val:
        :return:
        """
        coeff_cov = self.coefficients.posterior_cov
        coeff_mu = self.coefficients.posterior_mu
        V = self.polyobj.get_poly(x_test).T
        m1, S11 = self.get_poly_posterior(x_test)
        m2 = coeff_mu[0]
        S12 = (V @ coeff_cov[0]).reshape(-1, 1)
        S21 = S12.T
        S22 = coeff_cov[0, 0]

        b = m1 - 1. / S22 * S12 * m2
        c = 1. / S22 * S12

        mean = b + mean_val * c
        cov = S11 - S12 @ S21 / S22 + mean_var * c @ c.T

        return mean, cov

    def cokriging(self, lf_coeff_mean, lf_coeff_cov, rho, lf_x_train, lf_y_train, hf_x_train,
                  hf_y_train, hf_x_test):
        P_l = self.polyobj.get_poly(lf_x_train).T
        P_h = self.polyobj.get_poly(hf_x_train).T
        P_h_star = self.polyobj.get_poly(hf_x_test).T

        hf_coeff_mean = self.coefficients.posterior_mu
        hf_coeff_cov = self.coefficients.posterior_cov

        mu1 = P_l @ lf_coeff_mean.reshape(-1, 1)
        mu2 = P_h @ hf_coeff_mean.reshape(-1, 1)
        mu3 = P_h_star @ hf_coeff_mean.reshape(-1, 1)

        A11 = P_l @ lf_coeff_cov @ P_l.T
        A12 = rho * P_l @ lf_coeff_cov @ P_h.T
        A21 = A12.T
        A22 = P_h @ hf_coeff_cov @ P_h.T

        A13 = rho * P_l @ lf_coeff_cov @ P_h_star.T
        A23 = P_h @ hf_coeff_cov @ P_h_star.T
        A31 = A13.T
        A32 = A23.T
        A33 = P_h_star @ hf_coeff_cov @ P_h_star.T

        m1 = np.hstack([mu1.reshape(-1), mu2.reshape(-1)])
        m2 = mu3.copy().reshape(-1)

        K11 = np.vstack([np.hstack([A11, A12]), np.hstack([A21, A22])])
        K12 = np.vstack([A13, A23])
        K21 = np.hstack([A31, A32])
        K22 = A33.copy()

        y_train = np.hstack([lf_y_train.reshape(-1), hf_y_train.reshape(-1)])
        # print(m2.shape)
        # print((K21 @ np.linalg.solve(K11, y_train - m1)).shape)
        mean = m2 + K21 @ np.linalg.solve(K11, y_train - m1)
        cov = K22 - K21 @ np.linalg.inv(K11) @ K12

        return mean, cov

    # def condition_on_mean(self, scalar_mu, scalar_var, poly_cov, Xin, Xout, Yin):
    #
    #     #self.polyobj.get_poly(self.quad_pts)
    #     #int_P = np.zeros((self.P.shape[0], 1))
    #     #for i in range(0, self.P.shape[0]):
    #     #    int_P[i] = self.P[i,:] @ self.quad_wts
    #
    #     def get_sub_covar_matrix(plane, x_1=None, x_2=None):
    #         pt, wt = self.polyobj.get_points_and_weights()
    #         if plane == 11:
    #             return np.array([scalar_var]).reshape(1,1)
    #         elif plane == 22:
    #             P1 = self.polyobj.get_poly(x_1)
    #             P2 = self.polyobj.get_poly(x_2)
    #             return P1.T @ poly_cov @ P2
    #         elif plane == 12:
    #             P2 = self.polyobj.get_poly(x_2)
    #             self.P = self.polyobj.get_poly(pt)
    #             int_P = np.zeros((self.P.shape[0], 1))
    #             for i in range(0, self.P.shape[0]):
    #                 int_P[i] = self.P[i,:] @ wt
    #             return int_P.T @ poly_cov @ P2
    #         elif plane == 21:
    #             P1 = self.polyobj.get_poly(x_1)
    #             self.P = self.polyobj.get_poly(pt)
    #             int_P = np.zeros((self.P.shape[0], 1))
    #             for i in range(0, self.P.shape[0]):
    #                 int_P[i] = self.P[i,:] @ wt
    #             return P1.T @ poly_cov @ int_P
    #
    #     def covariance_matrix(D1, D2):
    #         K11 = get_sub_covar_matrix(11)
    #         K12 = get_sub_covar_matrix(12, D1, D2)
    #         K21 = get_sub_covar_matrix(21, D1, D2)
    #         K22 = get_sub_covar_matrix(22, D1, D2)
    #         top = np.concatenate([K11, K12], axis=1)
    #         bottom = np.concatenate([K21, K22], axis=1)
    #         return np.concatenate([top, bottom], axis=0)
    #
    #     C = np.linalg.inv( covariance_matrix(Xin, Xin) )
    #     R = covariance_matrix(Xout, Xin)
    #     RT = covariance_matrix(Xin, Xout)
    #     Cxx = covariance_matrix(Xout, Xout)
    #     print(Xin.shape, Xout.shape, C.shape, R.shape, Cxx.shape)
    #     mu_combined = R @ C @  np.vstack([scalar_mu, Yin])
    #     cov_combined = Cxx - (R @ C @ RT)
    #     return mu_combined, cov_combined

    def coregionalise(self, alpha, another_cov, Xin, Xout, Yin):
        """
        For the coregional model, we assume the size of X dims + 1, where
        the last column corresponds to which polynomial the remaining columns
        are associated with.
        """
        dims = self.polyobj.dimensions

        def get_sub_covar_matrix(plane, x_1, x_2):
            P1 = self.polyobj.get_poly(x_1)
            P2 = self.polyobj.get_poly(x_2)
            if plane == 11:
                return P1.T @ self.coefficients.posterior_cov @ P2
            elif plane == 22:
                return P1.T @ another_cov @ P2
            elif plane == 12:
                return  alpha *  (P1.T @ self.coefficients.posterior_cov  @ P2)
            elif plane == 21:
                return alpha * ( P1.T @ self.coefficients.posterior_cov  @ P2 )

        def covariance_matrix(D1, D2):
            first_poly_X1 = (D1[:,dims] == 0)
            second_poly_X1 = (D1[:, dims] == 1)
            x1 = D1[first_poly_X1, 0:dims]
            x2 = D1[second_poly_X1, 0:dims]

            first_poly_X2 = (D2[:,dims] == 0)
            second_poly_X2 = (D2[:, dims] == 1)
            x1_prime = D2[first_poly_X2, 0:dims]
            x2_prime = D2[second_poly_X2, 0:dims]

            K11 = get_sub_covar_matrix(11, x1, x1_prime)
            K12 = get_sub_covar_matrix(12, x1, x2_prime)
            K21 = get_sub_covar_matrix(21, x2, x1_prime)
            K22 = get_sub_covar_matrix(22, x2, x2_prime)

            top = np.concatenate([K11, K12], axis=1)
            bottom = np.concatenate([K21, K22], axis=1)
            return np.concatenate([top, bottom], axis=0)

        C = np.linalg.inv( covariance_matrix(Xin, Xin) )
        R = covariance_matrix(Xout, Xin)
        RT = covariance_matrix(Xin, Xout)
        Cxx = covariance_matrix(Xout, Xout)
        mu_combined = R @ C @  Yin
        cov_combined = Cxx - (R @ C @ RT)
        return mu_combined, cov_combined

def marginal_likelihood(pb, x_star, x_train, y_train):
    """
    Returns the negative log marginal likelihood and its gradient.
    """
    x = np.vstack([x_train, x_star]) # input data "x"
    t_star = float( pb.polyobj.get_polyfit(np.array(x_star).reshape(1, 1) ))
    t = np.vstack([y_train, t_star])
    N = len(x)
    P = pb.polyobj.get_poly(x) # Vandermonde type matrix
    mu_m = P.T @ pb.coefficients.prior_mu # Marginal likelihood mean
    Sigma_m = pb.coefficients.sigma_data**2 * np.eye(N) + \
                        (P.T @ pb.coefficients.prior_cov @ P) # Marginal likelihood cov

    # Gradient calculations!
    dX = pb.polyobj.get_poly_grad(x).T
    dt = pb.polyobj.get_polyfit_grad(x).reshape(len(t), 1)
    dX_dxstar = np.vstack([ np.zeros((dX.shape[0]-1, dX.shape[1])), dX[-1,:]])
    dt_dxstar = np.vstack([np.zeros((len(t)-1, 1)), dt[-1] ])
    dmum_dxstar = dX_dxstar @ pb.coefficients.prior_mu
    dSigmam_dxstar =  (P.T @ pb.coefficients.prior_cov @ dX_dxstar.T) \
                + (dX_dxstar @ pb.coefficients.prior_cov @ P)
    dlogdetSigmam_dxstar = np.trace(np.linalg.inv(Sigma_m) @ dSigmam_dxstar)
    dinvSigmam_dxstar = - np.linalg.inv(Sigma_m) @ dSigmam_dxstar @ np.linalg.inv(Sigma_m)

    # -ve log marginal likelihood
    v = (t - mu_m).T @ np.linalg.inv(Sigma_m) @ (t - mu_m)
    dv = (dt_dxstar - dmum_dxstar).T @ np.linalg.inv(Sigma_m) @ (t - mu_m) + \
         (t - mu_m).T @ dinvSigmam_dxstar @ (t - mu_m) + \
         (t - mu_m).T @ np.linalg.inv(Sigma_m) @ (dt_dxstar - dmum_dxstar)
    R = 0.5 * N * np.log(2. * np.pi) + N * 0.5 * np.log(np.linalg.det(Sigma_m)) + 0.5 * v
    dR_dx = N * 0.5 * dlogdetSigmam_dxstar + 0.5 * dv
    return float(R), dR_dx.reshape(-1)


def posterior_variance_opt(pb, x_star):
    """
    Returns the posterior variance and its
    """
    # x_star = deepcopy( np.array([x_star]) )
    K = len(x_star)
    P_pred = pb.polyobj.get_poly(x_star)
    f = P_pred.T @ pb.coefficients.posterior_cov @ P_pred

    dP_dxstar = pb.polyobj.get_poly_grad(x_star)
    if isinstance(dP_dxstar, list):
        dP_dxstar = np.hstack(dP_dxstar)
    df = 2.0 * P_pred.T @ pb.coefficients.posterior_cov @ dP_dxstar
    # print(x_star, f, df)
    # return -float(f), -df
    return -float(f), -df.reshape(-1)

def det_of_posterior(pb, x_star):
    current_post_cov = pb.coefficients.posterior_cov
    inv_current_post_cov = np.linalg.inv(current_post_cov)

    t0 = time()
    v_star = pb.polyobj.get_poly(x_star).reshape(-1, 1)
    sigma_data = pb.coefficients.sigma_data
    A = 1. / sigma_data ** 2 * v_star @ v_star.T + inv_current_post_cov

    t1 = time()
    # roll = np.random.uniform(0, 1)
    # thres = 0.99
    # if roll > thres:
    #     print('---------------')
    #     print(t1 - t0)
    dv_star = pb.polyobj.get_poly_grad(x_star)
    d = len(x_star)
    if not isinstance(dv_star, list):
        assert d == 1
        dv_star = [dv_star]

    df = []
    t2 = time()
    # if roll > thres:
    #     print(t2 - t1)
    for vec in dv_star:
        vec = vec.reshape(-1, 1)
        dAdxi = 1. / sigma_data ** 2 * (vec @ v_star.T + v_star @ vec.T)
        dfdxi = -np.trace(np.linalg.inv(A) @ dAdxi)
        df.append(dfdxi)
    df = np.array(df)

    t3 = time()
    # if roll > thres:
    #     print(t3 - t2)
    sign, logdet = np.linalg.slogdet(A)
    f = -sign * logdet
    t4 = time()
    # if roll > thres:
    #     print(t4 - t3)
    #     print('---------------')

    return f, df
