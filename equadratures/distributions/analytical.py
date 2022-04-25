""" The Analytical distribution"""
from equadratures.distributions.template import Distribution
from equadratures.distributions.uniform import Uniform 
# from equadratures import Poly, Basis
import numpy as np
import scipy.stats as stats
RECURRENCE_PDF_SAMPLES = 50000
ORDER_LIMIT = 5000
RECURRENCE_PDF_SAMPLES = 50000
QUADRATURE_ORDER_INCREMENT = 80
class Analytical(Distribution):
    """ The class offers a template to input bespoke weight (probability density) functions. The resulting weight function can be given to :class:`~equadratures.parameter.Parameter` to create a bespoke analytical or data-driven parameter.

    Parameters
    ----------
    weight_function : ~collections.abc.Callable,numpy.ndarray
        A callable function.
    support : list, optional
        Lower and upper bounds of the weight respectively. Values such as ``-inf`` or ``inf`` are not acceptable.
    pdf : bool, optional
        If set to ``True``, then the weight_function is assumed to be normalised to integrate to unity. Otherwise,
        the integration constant is computed and used to normalise weight_function.
    mean : float, optional
        User-defined mean for distribution. When provided, the code does not compute the mean of the weight_function over its support.
    variance : float, optional 
        User-defined variance for distribution. When provided, the code does not compute the variance of the weight_function over its support.

    Example
    -------
    Analytical weight functions
        >>> # exp(-x)/sqrt(x)
        >>> pdf_1 = Weight(lambda x: np.exp(-x)/ np.sqrt(x), [0.00001, -np.log(1e-10)], 
        >>>        pdf=False)
        >>> 
        >>> # A triangular distribution
        >>> a = 3.
        >>> b = 6.
        >>> c = 4.
        >>> mean = (a + b + c)/3.
        >>> var = (a**2 + b**2 + c**2 - a*b - a*c - b*c)/18.
        >>> pdf_2 = Weight(lambda x : 2*(x-a)/((b-a)*(c-a)) if (a <= x < c) 
        >>>                         else( 2/(b-a) if (x == c) 
        >>>                         else( 2*(b-x)/((b-a)*(b-c)))), 
        >>>             support=[a, b], pdf=True)
        >>> 
        >>> # Passing to Parameter
        >>> param = Parameter(distribution='analytical', weight_function=pdf_2, order=2)

    Data driven weight functions
        >>> # Constructing a kde based on given data, using Rilverman's rule for bandwidth selection
        >>> pdf_2 = Weight( stats.gaussian_kde(data, bw_method='silverman'), 
        >>>        support=[-3, 3.2])
        >>> 
        >>> # Passing to Parameter
        >>> param = Parameter(distribution='analytical', weight_function=pdf, order=2)

    """
    def __init__(self, **kwargs):
        first_arg = ['weight-function', 'weight_function', 'weight', 'function']
        second_arg = ['pdf', 'PDF', 'probability']
        third_arg = ['lower', 'low', 'bottom']
        fourth_arg = ['upper','up', 'top']  
        fifth_arg = ['order', 'orders', 'degree', 'degrees']
        sixth_arg = ['endpoints', 'endpoint']
        self.name = 'analytical'
        self.lower = None 
        self.upper = None
        self.order = 2
        self.endpoints = 'none'
        for key, value in kwargs.items():
            if first_arg.__contains__(key):
                self.weight_function = value 
            if second_arg.__contains__(key):
                self.pdf = value 
            if third_arg.__contains__(key):
                self.lower = value 
            if fourth_arg.__contains__(key):
                self.upper = value 
            if fifth_arg.__contains__(key):
                self.order = value
            if sixth_arg.__contains__(key):
                self.endpoints = value

       
        self.mean = weight_function.mean
        self.variance = weight_function.variance
        self.bounds = weight_function.support
        self.x_range_for_pdf = weight_function.x_range_for_pdf

        analytical = stats.rv_discrete(name='analytical', values=(self.x_range_for_pdf, p))
        
        self.parent = beta(self.shape_A, self.shape_B, loc=loc, scale=scale)
        self.mean, self.variance, self.skewness, self.kurtosis = beta.stats(self.shape_A, self.shape_B, loc=loc, scale=scale, moments='mvsk')
        self.x_range_for_pdf = np.linspace(self.lower, self.upper, RECURRENCE_PDF_SAMPLES)
        super().__init__(name=self.name, \
                        lower=self.lower, \
                        upper=self.upper, \
                        mean=self.mean, \
                        variance=self.variance, \
                        skewness=self.skewness, \
                        kurtosis=self.kurtosis, \
                        x_range_for_pdf=self.x_range_for_pdf, \
                        order=self.order, \
                        endpoints=self.endpoints, \
                        scipyparent=self.parent)

    def get_description(self):
        """ A destription of Analytical distribution.

            :param Analytical self:
                An instance of Analytical class.
            :return:
                A string describing the Analytical distribution.
        """
        text = "is a Analytical distribution defined over a support from "+str(self.lower)+" to "+str(self.upper)+". \
            It has a mean value equal to "+str(self.mean)+" and a variance equal to "+str(self.variance)+"."
        return text

    def get_pdf(self, points=None):
        """ A Analytical probability density function.

            :param Analytical self:
                An instance of Analytical class.
            :param points:
                An array of points in which the probability density function needs to be calculated.
            :return:
                Probability density values along the support of Analytical distribution.
            ** Notes **
            To obtain a probability density function from finite samples, this function uses kerne density estimation (with Gaussian kernel).
        """
        return self.weight_function.get_pdf(points)

    def get_cdf(self, points=None):
        y = self.get_pdf()
        summ = np.sum(y)
        p = np.array(y/summ)
        analytical = stats.rv_discrete(name='analytical', values=(self.x_range_for_pdf, p))
        return analytical.cdf(points)

    def get_recurrence_coefficients(self, order):
        """
        Recurrence coefficients for the Analytical distribution.

        :param Analytical self:
            An instance of Analytical class.
        :param array order:
            The order of the recurrence coefficients desidered.
        :return:
            Recurrence coefficients associated with the Analytical distribution.
        """
        #x = np.linspace(self.lower, self.upper, RECURRENCE_PDF_SAMPLES)
        w = self.data

        # Allocate memory for recurrence coefficients
        order = int(order)+1
        w = w / np.sum(w)
        ab = np.zeros((order+1,2))

        # Negate "zero" components
        nonzero_indices = []
        for i in range(0, len(w)):
            if w[i] != 0:
                nonzero_indices.append(i)

        ncap = len(nonzero_indices)
        x = self.x_range_for_pdf[nonzero_indices] # only keep entries at the non-zero indices!
        w = w[nonzero_indices]
        s = np.sum(w)

        temp = w/s
        ab[0,0] = np.dot(x, temp.T)
        ab[0,1] = s

        if order == 1:
            return ab

        p1 = np.zeros((1, ncap))
        p2 = np.ones((1, ncap))

        for j in range(0, order):
            p0 = p1
            p1 = p2
            p2 = ( x - ab[j,0] ) * p1 - ab[j,1] * p0
            p2_squared = p2**2
            s1 = np.dot(w, p2_squared.T)
            inner = w * p2_squared
            s2 = np.dot(x, inner.T)
            ab[j+1,0] = s2/s1
            ab[j+1,1] = s1/s
            s = s1
        return ab

    def get_icdf(self, xx):
        """
        A Analytical inverse cumulative distribution function.

        :param Analytical self:
            An instance of Analytical class.
        :param array xx:
            An array of points in which the inverse cumulative density function needs to be evaluated.
        :return:
            Inverse cumulative density function values of the Analytical distribution.
        """
        y = self.get_pdf()
        summ = np.sum(y)
        p = np.array(y/summ)
        analytical = stats.rv_discrete(name='analytical', values=(self.x_range_for_pdf, p))
        return analytical.ppf(xx)

    def _evaluate_pdf(self, x):
        x = np.array(x)
        pdf_values = np.zeros((x.shape[0]))
        for i in range(0, x.shape[0]):
            pdf_values[i] = self.weight_function(x[i])
        return pdf_values

    def get_pdf(self, points=None):
        """ Returns the pdf associated with the distribution.

        Parameters
        ----------
        points : numpy.ndarray, optional
            Array of points to evaluate pdf at.

        Returns
        -------
        numpy.ndarray
            Array with shape ( len(points),1 ) containing the probability distribution.

        """
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
        param = Uniform(lower=self.lower, upper=self.upper,order=order)
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
