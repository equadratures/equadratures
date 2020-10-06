import numpy as np
from scipy.optimize import minimize
from copy import deepcopy

class Coeffbayes(object):
    """
    A class for coefficients.
    """
    def __init__(self, prior_mu, prior_cov, sigma_data=1e-7):
        self.prior_mu = prior_mu
        self.prior_cov = prior_cov
        self.inv_prior_cov = np.linalg.inv(self.prior_cov)
        self.sigma_data = sigma_data # nugget / uncertainty in model output!

    def compute_posterior_coefficients(self, polyobj, x_train, y_train):
        assert x_train.shape[0] == y_train.shape[0]
        P = polyobj.get_poly(x_train)
        self.posterior_cov = np.linalg.inv( 1./self.sigma_data**2 * P @ P.T + self.inv_prior_cov )
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
        P = self.polyobj.get_poly(x_test)
        mu = self.coefficients.prior_mu
        Sigma = self.coefficients.prior_cov
        polymu = P.T @ mu.reshape(len(mu), 1)
        polycov = P.T @ Sigma @ P
        return polymu, polycov

    def get_poly_posterior(self, x_test, coeff_mu=None, coeff_var=None):
        P = self.polyobj.get_poly(x_test)
        if coeff_mu is None:
            mu = self.coefficients.posterior_mu
        if coeff_var is None:
            Sigma = self.coefficients.posterior_cov
        polymu = P.T @ mu.reshape(len(mu), 1)
        polycov = P.T @ Sigma @ P
        return polymu, polycov

    def get_monte_carlo_moments(self):
        """
        Sample the polynomial with random x's
        """
        xx = np.random.rand(30000,1)*2. - 1.
        mu_samples, cov_samples = self.get_poly_posterior(xx)
        std_samples = np.sqrt(np.diag(cov_samples))
        return mu_samples, std_samples

    def get_poly_chaos_moments(self):
        """
        Returns the polynomial chaos mean and variance using the mean posterior polynomial.
        """
        def polyfunc(x):
            P = self.polyobj.get_poly(x)
            mu, _ = self.get_poly_posterior(x)
            return mu

        # Create a polynomial instance
        param = self.polyobj.parameters
        basis = self.polyobj.basis
        poly = eq.Poly(param, basis, method=self.polyobj.method)
        poly.set_model(polyfunc)
        return poly.get_mean_and_variance()

    def get_new_input_location(self, method='variance', x_train=None, y_train=None):
        """
        Returns a new location based on an optimisation. Note that the
        training data provided to the Coefficients is used. This can
        be overwritten if required.

        For multi-fidelity useage, typically the high-fidelity data is used.
        """
        if x_train is None:
            x_train = self.coefficients.x_train
            y_train = self.coefficients.y_train

        if method.lower() is 'variance':
            function = lambda value: posterior_variance_opt(value)
        else:
            function = lambda value: marginal_likelihood(value)

        def marginal_likelihood(x_star):
            """
            Returns the negative log marginal likelihood and its gradient.
            """
            x = np.vstack([x_train, x_star]) # input data "x"
            t_star = float( self.polyobj.get_polyfit(np.array(x_star).reshape(1, 1) ))
            t = np.vstack([y_train, t_star])
            N = len(x)
            P = self.polyobj.get_poly(x) # Vandermonde type matrix
            mu_m = P.T @ self.coefficients.prior_mu # Marginal likelihood mean
            Sigma_m = self.coefficients.sigma_data**2 * np.eye(N) + \
                                (P.T @ self.coefficients.prior_cov @ P) # Marginal likelihood cov

            # Gradient calculations!
            dX = self.polyobj.get_poly_grad(x).T
            dt = self.polyobj.get_polyfit_grad(x).reshape(len(t), 1)
            dX_dxstar = np.vstack([ np.zeros((dX.shape[0]-1, dX.shape[1])), dX[-1,:]])
            dt_dxstar = np.vstack([np.zeros((len(t)-1, 1)), dt[-1] ])
            dmum_dxstar = dX_dxstar @ self.coefficients.prior_mu
            dSigmam_dxstar =  (P.T @ self.coefficients.prior_cov @ dX_dxstar.T) \
                        + (dX_dxstar @ self.coefficients.prior_cov @ P)
            dlogdetSigmam_dxstar = np.trace(np.linalg.inv(Sigma_m) @ dSigmam_dxstar)
            dinvSigmam_dxstar = - np.linalg.inv(Sigma_m) @ dSigmam_dxstar @ np.linalg.inv(Sigma_m)

            # -ve log marginal likelihood
            v = (t - mu_m).T @ np.linalg.inv(Sigma_m) @ (t - mu_m)
            dv = (dt_dxstar - dmum_dxstar).T @ np.linalg.inv(Sigma_m) @ (t - mu_m) + \
                 (t - mu_m).T @ dinvSigmam_dxstar @ (t - mu_m) + \
                 (t - mu_m).T @ np.linalg.inv(Sigma_m) @ (dt_dxstar - dmum_dxstar)
            R = 0.5 * N * np.log(2. * np.pi) + N * 0.5 * np.log(np.linalg.det(Sigma_m)) + 0.5 * v
            dR_dx = N * 0.5 * dlogdetSigmam_dxstar + 0.5 * dv
            return float(R), float(dR_dx)

        def posterior_variance_opt(x_star):
            """
            Returns the posterior variance and its
            """
            x_star = deepcopy( np.array([x_star]) )
            K = len(x_star)
            P_pred = self.polyobj.get_poly(x_star.reshape(K, 1))
            dP_dxstar = self.polyobj.get_poly_grad(x_star)
            f = np.sqrt( P_pred.T @ self.coefficients.posterior_cov @ P_pred )
            df =  0.5 * ( P_pred.T @ self.coefficients.posterior_cov @ P_pred )**(-0.5) * \
                        ( P_pred.T @ Sigma_w @ dP_dxstar + dP_dxstar.T @ self.coefficients.posterior_cov @ P_pred)
            return float(f), float(df)

        myBounds = np.zeros(( self.polyobj.dimensions, 2))
        x0 = np.zeros((self.polyobj.dimensions, 1))
        for i in range(0, myBounds.shape[0]):
            myBounds[i, 0] = self.polyobj.parameters[i].lower
            myBounds[i, 1] = self.polyobj.parameters[i].upper
            x0[i] = self.polyobj.parameters[i].mean
        myOptions = {'maxiter': 150, 'disp':True, 'ftol':1e-7}
        opt_ret = minimize(fun=function, x0=x0, method='SLSQP', jac=True, \
                                    bounds=myBounds, options=myOptions)
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

    """"
    def condition_on_mean_and_var(self, scalar_mean_mu, scalar_mean_cov, scalar_var_mu, scalar_var_cov, \
                                  poly_cov, Xin, Xout, Yin):

        def get_sub_covar_matrix(plane, x_1=None, x_2=None):
            pt, wt = self.polyobj.get_points_and_weights()
            if plane == 11:
                return np.array([scalar_var]).reshape(1,1)
            elif plane == 22:
                P1 = self.polyobj.get_poly(x_1)
                P2 = self.polyobj.get_poly(x_2)
                return P1.T @ poly_cov @ P2
            elif plane == 12:
                P2 = self.polyobj.get_poly(x_2)
                self.P = self.polyobj.get_poly(pt)
                int_P = np.zeros((self.P.shape[0], 1))
                for i in range(0, self.P.shape[0]):
                    int_P[i] = self.P[i,:] @ wt
                return int_P.T @ poly_cov @ P2
            elif plane == 21:
                P1 = self.polyobj.get_poly(x_1)
                self.P = self.polyobj.get_poly(pt)
                int_P = np.zeros((self.P.shape[0], 1))
                for i in range(0, self.P.shape[0]):
                    int_P[i] = self.P[i,:] @ wt
                return P1.T @ poly_cov @ int_P

        def covariance_matrix(D1, D2):
            K11 = get_sub_covar_matrix(11)
            K12 = get_sub_covar_matrix(12, D1, D2)
            K13 = get_sub_covar_matrix(13, D1, D2)

            K21 = get_sub_covar_matrix(21, D1, D2)
            K22 = get_sub_covar_matrix(22, D1, D2)
            K23 = get_sub_covar_matrix(23, D1, D2)

            K31 = get_sub_covar_matrix(31, D1, D2)


            top = np.concatenate([K11, K12], axis=1)
            bottom = np.concatenate([K21, K22], axis=1)
            return np.concatenate([top, bottom], axis=0)

        C = np.linalg.inv( covariance_matrix(Xin, Xin) )
        R = covariance_matrix(Xout, Xin)
        RT = covariance_matrix(Xin, Xout)
        Cxx = covariance_matrix(Xout, Xout)
        print(Xin.shape, Xout.shape, C.shape, R.shape, Cxx.shape)
        mu_combined = R @ C @  np.vstack([scalar_mu, Yin])
        cov_combined = Cxx - (R @ C @ RT)
        return mu_combined, cov_combined

    """
    def condition_on_mean(self, scalar_mu, scalar_var, poly_cov, Xin, Xout, Yin):

        #self.polyobj.get_poly(self.quad_pts)
        #int_P = np.zeros((self.P.shape[0], 1))
        #for i in range(0, self.P.shape[0]):
        #    int_P[i] = self.P[i,:] @ self.quad_wts

        def get_sub_covar_matrix(plane, x_1=None, x_2=None):
            pt, wt = self.polyobj.get_points_and_weights()
            if plane == 11:
                return np.array([scalar_var]).reshape(1,1)
            elif plane == 22:
                P1 = self.polyobj.get_poly(x_1)
                P2 = self.polyobj.get_poly(x_2)
                return P1.T @ poly_cov @ P2
            elif plane == 12:
                P2 = self.polyobj.get_poly(x_2)
                self.P = self.polyobj.get_poly(pt)
                int_P = np.zeros((self.P.shape[0], 1))
                for i in range(0, self.P.shape[0]):
                    int_P[i] = self.P[i,:] @ wt
                return int_P.T @ poly_cov @ P2
            elif plane == 21:
                P1 = self.polyobj.get_poly(x_1)
                self.P = self.polyobj.get_poly(pt)
                int_P = np.zeros((self.P.shape[0], 1))
                for i in range(0, self.P.shape[0]):
                    int_P[i] = self.P[i,:] @ wt
                return P1.T @ poly_cov @ int_P

        def covariance_matrix(D1, D2):
            K11 = get_sub_covar_matrix(11)
            K12 = get_sub_covar_matrix(12, D1, D2)
            K21 = get_sub_covar_matrix(21, D1, D2)
            K22 = get_sub_covar_matrix(22, D1, D2)
            top = np.concatenate([K11, K12], axis=1)
            bottom = np.concatenate([K21, K22], axis=1)
            return np.concatenate([top, bottom], axis=0)

        C = np.linalg.inv( covariance_matrix(Xin, Xin) )
        R = covariance_matrix(Xout, Xin)
        RT = covariance_matrix(Xin, Xout)
        Cxx = covariance_matrix(Xout, Xout)
        print(Xin.shape, Xout.shape, C.shape, R.shape, Cxx.shape)
        mu_combined = R @ C @  np.vstack([scalar_mu, Yin])
        cov_combined = Cxx - (R @ C @ RT)
        return mu_combined, cov_combined

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
