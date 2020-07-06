"Definition of a probability distribution."
from equadratures.parameter import Parameter
from equadratures.basis import Basis
from equadratures.poly import Poly, evaluate_model
import numpy as np
ORDER_LIMIT = 5000
RECURRENCE_PDF_SAMPLES = 50000
QUADRATURE_ORDER_INCREMENT = 80
class Weight(object):
    """
    The class offers a template to input bespoke weight (probability density) functions.

    :param lambda weight_function: A function call.
    :param list support: Lower and upper bounds of the weight respectively.
    :param bool pdf: If set to ``True``, then the weight_function is assumed to be normalised to integrate to unity. Otherwise,
        the integration constant is computed and used to normalise weight_function.
    :param float mean: User-defined mean for distribution. When provided, the code does not compute the mean of the weight_function over its support.
    :param float variance: User-defined variance for distribution. When provided, the code does not compute the variance of the weight_function over its support.

    **Sample constructor initialisations**::

        import numpy as np
        from equadratures import *

        pdf = Weight(lambda x: exp(-x)/ np.sqrt(x), [0.00001, -np.log(1e-10)], pdf=False)
    """
    def __init__(self, weight_function, support, pdf=False, mean=None, variance=None):
        self.weight_function = weight_function
        self.pdf = pdf
        self.support = support
        self.lower = self.support[0]
        self.upper = self.support[1]
        if self.upper <= self.lower:
            raise(ValueError, 'The lower bound must be less than the upper bound in the support.')
        if self.lower == -np.inf:
            raise(ValueError, 'The lower bound cannot be negative infinity.')
        if self.upper == np.inf:
            raise(ValueError, 'The upper bound cannot be infinity.')
        self._verify_probability_density()
        self.x_range_for_pdf = np.linspace(self.lower, self.upper, RECURRENCE_PDF_SAMPLES)
        self.mean = mean
        self.variance = variance
        self.data = self.get_pdf()
        if self.mean is None:
            self._set_mean()
        if self.variance is None:
            self._set_variance()

    def _evaluate_pdf(self, x):
        x = np.array(x)
        pdf_values = np.zeros((x.shape[0]))
        for i in range(0, x.shape[0]):
            pdf_values[i] = self.weight_function(x[i])
        return pdf_values

    def get_pdf(self, points=None):
        if points is None:
            return self._evaluate_pdf(self.x_range_for_pdf) * self.integration_constant
        else:
            return self._evaluate_pdf(points) * self.integration_constant

    def _verify_probability_density(self):
        integral, _ = self._iterative_quadrature_computation(self.weight_function)
        if (np.abs(integral - 1.0) >= 1e-2) or (self.pdf is False):
            self.integration_constant = 1.0/integral
        elif (np.abs(integral - 1.0) < 1e-2) or (self.pdf is True):
            self.integration_constant = 1.0

    def _get_quadrature_points_and_weights(self, order):
        param = Parameter(distribution='uniform',lower=self.lower, upper=self.upper,order=order)
        basis = Basis('univariate')
        poly = Poly(method='numerical-integration',parameters=param,basis=basis)
        points, weights = poly.get_points_and_weights()
        return points, weights * (self.upper - self.lower)

    def _set_mean(self):
        # Modified integrand for estimating the mean
        mean_integrand = lambda x: x * self.weight_function(x) * self.integration_constant
        self.mean, self._mean_quadrature_order = self._iterative_quadrature_computation(mean_integrand)

    def _iterative_quadrature_computation(self, integrand, quadrature_order_output=True):
        # Keep increasing the order till we reach ORDER_LIMIT
        quadrature_error = 500.0
        quadrature_order = 0
        integral_before = 10.0
        while quadrature_error >= 1e-6:
            quadrature_order += QUADRATURE_ORDER_INCREMENT
            pts, wts = self._get_quadrature_points_and_weights(quadrature_order)
            integral = float(np.dot(wts, evaluate_model(pts, integrand)))
            quadrature_error = np.abs(integral - integral_before)
            integral_before = integral
            if quadrature_order >= ORDER_LIMIT:
                raise(RuntimeError, 'Even with '+str(ORDER_LIMIT+1)+' points, an error in the mean of '+str(1e-4)+'cannot be obtained.')
        if quadrature_order_output is True:
            return integral, quadrature_order
        else:
            return integral

    def _set_variance(self):
        # Modified integrand for estimating the mean
        variance_integrand = lambda x: (x  - self.mean)**2 * self.weight_function(x) * self.integration_constant
        self.variance, self._variance_quadrature_order = self._iterative_quadrature_computation(variance_integrand)
