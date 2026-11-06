"""The Analytical distribution."""
import numpy as np
import scipy.stats as stats

from equadratures.distributions.recurrence_utils import custom_recurrence_coefficients
from equadratures.distributions.template import Distribution


class Analytical(Distribution):
    """A distribution defined by a bespoke weight function.

    The weight function is an instance of :class:`~equadratures.weight.Weight`.
    It is used for both custom analytical and data-driven parameters.

    Parameters
    ----------
    weight_function : Weight
        An instance of the Weight class, defining the probability density function and its support.
    order : int, optional
        Order of the parameter. Default: 2.
    endpoints : str, optional
        Whether to include the endpoints of the support in the quadrature rule. Default: ``'none'``.

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
        fifth_arg = ['order', 'orders', 'degree', 'degrees']
        sixth_arg = ['endpoints', 'endpoint']
        self.name = 'analytical'
        self.weight_function = None
        self.order = 2
        self.endpoints = 'none'
        for key, value in kwargs.items():
            if key in first_arg:
                self.weight_function = value
            if key in fifth_arg:
                self.order = value
            if key in sixth_arg:
                self.endpoints = value
        if self.weight_function is None:
            raise ValueError(
                'An Analytical distribution requires a weight_function (an instance of Weight).'
            )

        super().__init__(
            name=self.name,
            lower=self.weight_function.lower,
            upper=self.weight_function.upper,
            mean=self.weight_function.mean,
            variance=self.weight_function.variance,
            x_range_for_pdf=self.weight_function.x_range_for_pdf,
            order=self.order,
            endpoints=self.endpoints,
        )

    def get_description(self):
        """ A destription of Analytical distribution.

            :param Analytical self:
                An instance of Analytical class.
            :return:
                A string describing the Analytical distribution.
        """
        text = (
            'is a Analytical distribution defined over a support from ' + str(self.lower)
            + ' to ' + str(self.upper)
            + '. It has a mean value equal to ' + str(self.mean)
            + ' and a variance equal to ' + str(self.variance) + '.'
        )
        return text

    def get_pdf(self, points=None):
        """ The probability density function of the Analytical distribution.

            :param Analytical self:
                An instance of Analytical class.
            :param points:
                An array of points at which the probability density function needs to be calculated.
            :return:
                Probability density values at the points. If no points are given, a tuple of
                the support points and the density values along them.
        """
        if points is None:
            return self.x_range_for_pdf, self.weight_function.get_pdf()
        return self.weight_function.get_pdf(points)

    def _get_discrete_distribution(self):
        y = self.weight_function.get_pdf()
        return stats.rv_discrete(
            name='analytical', values=(self.x_range_for_pdf, y / np.sum(y))
        )

    def get_cdf(self, points=None):
        """ The cumulative distribution function of the Analytical distribution.

            :param Analytical self:
                An instance of Analytical class.
            :param points:
                Points at which the cumulative distribution function needs to be calculated.
            :return:
                Cumulative distribution values at the points. If no points are given, a tuple of
                the support points and the values along them.
        """
        if points is None:
            return self.x_range_for_pdf, self._get_discrete_distribution().cdf(
                self.x_range_for_pdf
            )
        return self._get_discrete_distribution().cdf(points)

    def get_icdf(self, xx):
        """ The inverse cumulative distribution function of the Analytical distribution.

            :param Analytical self:
                An instance of Analytical class.
            :param array xx:
                Points at which the inverse cumulative distribution function needs to be evaluated.
            :return:
                Inverse cumulative distribution values of the Analytical distribution.
        """
        return self._get_discrete_distribution().ppf(xx)

    def get_samples(self, m=None):
        """ Generates samples from the Analytical distribution.

            :param Analytical self:
                An instance of Analytical class.
            :param integer m:
                Number of random samples. If not provided, a default of 5e5 is assumed.
            :return:
                An array of m samples.
        """
        number = 500000 if m is None else m
        return self.get_icdf(np.random.uniform(size=number))

    def get_recurrence_coefficients(self, order):
        """ Recurrence coefficients for the Analytical distribution.

            :param Analytical self:
                An instance of Analytical class.
            :param array order:
                The order of the recurrence coefficients desired.
            :return:
                Recurrence coefficients associated with the Analytical distribution.
        """
        # The density is already stored on the weight.
        # Recomputing it is slow for data-driven weights.
        return custom_recurrence_coefficients(
            self.x_range_for_pdf, self.weight_function.data, order
        )
