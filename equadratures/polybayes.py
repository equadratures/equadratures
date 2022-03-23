import numpy as np
from equadratures.poly import Poly
from equadratures import Parameter
from equadratures.basis import Basis


class Polybayes(object):
    """ This class defines a Polybayes object. It is used to fit a Bayesian orthogonal polynomial. A key assumption we make is that the prior and posterior coefficients, predictions and data are Gaussian distributed.

    Parameters
    ----------
    parameters : list of ``Parameter``s, optional
        Input parameters defined by a list of ``Parameter`` instances.

    basis : ``Basis``, optional
        An instance of ``Basis`` that defines the polynomial basis functions.

    polyobj : ``Poly``, optional
        An instance of ``Poly`` that defines the input parameters and basis of the Bayesian polynomial. Either provide this or both of ``parameters`` and ``basis``.

    prior_mean : numpy.ndarray, optional
        A 1-D numpy array of length equal to the basis cardinality that is used for the prior mean of the coefficients. Defaults to an all zeros array.

    prior_cov : numpy.ndarray, optional
        A 2-D square numpy array of length equal to the basis cardinality that is used for the prior covariance of the coefficients. Defaults to an identity matrix.

    sigma_data : float, optional
        A positive number that defines the data noise standard deviation. Defaults to 1e-3.

    Examples
    --------
    Defining and fitting a Bayesian polynomial.
        >>>my_polybayes = eq.Polybayes(my_params, my_basis,    prior_mean=prior_coefficients_mean, prior_cov=prior_coefficients_cov, sigma_data=data_std)
        >>>my_polybayes.compute_posterior_coefficients(input_training_data, output_training_data)

    Evaluating the Bayesian polynomial (to obtain the mean, covariance and pointwise standard deviations of the predictions).
        >>>predicted_mean_test, predicted_cov_test = my_polybayes.get_posterior_fit(input_testing_data)
        >>>predicted_std_test = np.sqrt(np.diag(predicted_cov_test))

    References
    ----------
    1. Wong, C.Y., Seshadri, P., Duncan, A.B., Scillitoe, A.D., Parks, G.T. Prior-informed Uncertainty Modelling with Bayesian Polynomial Approximations. (https://arxiv.org/abs/2203.03508)
    """
    def __init__(self, parameters=None, basis=None, polyobj=None, prior_mean=None, prior_cov=None,
                 sigma_data=1e-3):
        if polyobj is None:
            my_polyobj = Poly(parameters, basis, method='least-squares')
        else:
            my_polyobj = polyobj
        cardinality = my_polyobj.basis.get_cardinality()
        if prior_mean is None:
            prior_mean = np.zeros(cardinality)
        else:
            prior_mean = prior_mean.reshape(-1)
        if prior_cov is None:
            prior_cov = np.eye(cardinality)
        self.coefficients = Coefficients(prior_mean, prior_cov, my_polyobj, sigma_data)
        self.polyobj = my_polyobj

    def compute_posterior_coefficients(self, training_inputs, training_outputs):
        """ Compute posterior coefficients given training data.

        Parameters
        ----------
        training_inputs : numpy.ndarray
            2-D array of training input data (where each row is one observation).

        training_outputs : numpy.ndarray
            1-D array of training output data.

        """
        self.coefficients.compute_posterior_coefficients(training_inputs, training_outputs)

    def get_posterior_fit(self, test_inputs, estimated_mean=None, estimated_mean_sigma=0.0):
        """ Yield predictions on test input data, optionally conditioned on information on the output (spatial) mean.

        Parameters
        ----------
        test_inputs : numpy.ndarray
            2-D array of test input data (where each row is one observation).

        estimated_mean : float, optional
            An estimate of the output spatial mean.

        estimated_mean_sigma: float, optional
            Estimate of the standard deviation representing the confidence in the estimated output spatial mean. Defaults to 0.

        Returns
        ----------
        numpy.ndarray
            A 1-D array giving the predicted mean of at the input points.

        numpy.ndarray
            A 2-D array giving the covariance of the predictions.
        """
        test_V = self.polyobj.get_poly(test_inputs).T
        coeff_mu, coeff_cov = self.coefficients.posterior_mu, self.coefficients.posterior_cov
        mean_fn_test = test_V @ coeff_mu
        cov_fn_test = test_V @ coeff_cov @ test_V.T
        if estimated_mean is None:
            return mean_fn_test, cov_fn_test
        else:
            m1 = mean_fn_test.reshape(-1, 1)
            m2 = coeff_mu[0]
            S11 = cov_fn_test
            S12 = (test_V @ coeff_cov[0]).reshape(-1, 1)
            S21 = S12.T
            S22 = coeff_cov[0, 0]

            b = m1 - 1. / S22 * S12 * m2
            c = 1. / S22 * S12

            mean_fn_aftercond = b + estimated_mean * c
            cov_fn_aftercond = S11 - S12 @ S21 / S22 + estimated_mean_sigma**2 * c @ c.T
            return mean_fn_aftercond.reshape(-1), cov_fn_aftercond

    def get_posterior_output_moments(self, N_samples=10000, N_bins=100):
        """ Yield predictions for the output spatial mean and standard deviation. For the spatial mean, the mean of the spatial mean and its standard deviation is given. For the spatial standard deviation, a binned estimate of the mode and its standard deviation are given instead.

        Parameters
        ----------
        N_samples : int, optional
            Number of Monte Carlo samples to estimate the output moments. Defaults to 10000.

        N_bins: int, optional
            Number of bins to use in a histogram of the output spatial standard deviation to estimate its mode.

        Returns
        ----------
        tuple
            The mean and standard deviation of the output spatial mean.

        tuple
            The binned mode and standard deviation of the output spatial standard deviation.
        """
        coeff_mu, coeff_cov = self.coefficients.posterior_mu, self.coefficients.posterior_cov
        coeff_samples = np.random.multivariate_normal(coeff_mu, coeff_cov, size=N_samples)
        mean_samples = coeff_samples[:, 0]
        std_samples = np.sqrt(np.sum(coeff_samples[:, 1:]**2, axis=1))
        std_hist = np.histogram(std_samples, bins=N_bins)
        std_mode = std_hist[1][np.argmax(std_hist[0])]
        return (np.mean(mean_samples), np.std(mean_samples)), \
               (std_mode, np.std(std_samples))


class Coefficients(object):
    """
    Private class for Bayesian polynomial coefficients.
    """
    def __init__(self, prior_mu, prior_cov, polyobj, sigma_data=1e-7):
        self.prior_mu = prior_mu
        self.prior_cov = prior_cov
        self.polyobj = polyobj
        self.inv_prior_cov = np.linalg.inv(self.prior_cov)
        self.sigma_data = sigma_data
        self.posterior_mu = prior_mu.copy()
        self.posterior_cov = prior_cov.copy()

    def compute_posterior_coefficients(self, training_inputs, training_outputs):
        polyobj = self.polyobj
        P = polyobj.get_poly(training_inputs)
        self.P = P
        A = 1./self.sigma_data**2 * P @ P.T + self.inv_prior_cov
        self.posterior_cov = np.linalg.inv(A)
        self.posterior_mu = self.posterior_cov @ (1. / self.sigma_data ** 2 * P @ training_outputs + \
                                                  self.inv_prior_cov @ self.prior_mu)
        self.x_train = training_inputs
        self.y_train = training_outputs

