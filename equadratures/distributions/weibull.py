"""The Weibull distribution."""
from equadratures.distributions.template import Distribution
from equadratures.distributions.recurrence_utils import custom_recurrence_coefficients
import numpy as np
from scipy.stats import weibull_min
RECURRENCE_PDF_SAMPLES = 8000

class Weibull(Distribution):
    """
    The class defines a Weibull object. It is the child of Distribution.

    :param double shape:
		Lower bound of the support of the Weibull distribution.
    :param double scale:
		Upper bound of the support of the Weibull distribution.
	:param numpy.ndarray data:
	    Data for which the distribution is to be set
    """
    def __init__(self, scale=None, shape=None, data=None):
        if shape is None:
            if data is None:
                self.shape = 1.0
                self.data  = None
            else:
                self.data = data
        else:
            self.shape = shape
            self.data = None

        if scale is None:
            if data is None:
                self.scale = 1.0
                self.data = None
            else:
                self.scale = None
                self.data = data
        else:
            self.scale = scale
            self.data = None

        self.bounds = np.array([0.0, np.inf])

        if self.data is not None:
            params=weibull_min.fit(self.data)
            self.scale=params[2]
            self.shape=params[0]
        
        if self.shape < 0 or self.scale < 0:
            raise ValueError('Invalid parameters in Weibull distribution. Shape and Scale should be positive.')
        self.parent = weibull_min(c=self.shape, scale=self.scale)
        self.mean, self.variance, self.skewness, self.kurtosis = self.parent.stats(moments='mvsk')
        self.x_range_for_pdf = np.linspace(0, self.scale*10, RECURRENCE_PDF_SAMPLES)

    def get_description(self):
        """
        A description of the Weibull distribution.

        :param Weibull self:
            An instance of the Weibull class.
        :return:
            A string describing the Weibull distribution.
        """
        text = "is a Weibull distribution with a shape parameter of "+str(self.shape)+" and a scale parameter of "+str(self.scale)
        return text

    def get_pdf(self, points=None):
        """
        A Weibull probability density function.

        :param Weibull self:
            An instance of the Weibull class.
        :param integer N:
            Number of points for defining the probability density function.
        """
        if points is not None:
            #w = self.shape/self.scale * (points/self.scale)**(self.shape-1.0) * np.exp(-1.0 * (points/self.scale)**self.shape )
            #return w
            return self.parent.pdf(points)
        else:
            raise ValueError( 'Please digit an input for getCDF method')
    def get_icdf(self, xx):
        """
        An inverse Weibull cumulative density function.

        :param Weibull self:
            An instance of the Weibull class.
        :param array xx:
            A numpy array of uniformly distributed samples between [0,1].
        :return:
            Inverse CDF samples associated with the Weibull distribution.
        """
        #return self.scale * (-np.log(1.0 - xx))**(1.0/self.shape)
        return self.parent.ppf(xx)
    def get_cdf(self, points=None):
        """
        A Weibull cumulative density function.

        :param Weibull self:
            An instance of the Weibull class.
        :param integer N:
            Number of points for defining the cumulative density function.
        :return:
            An array of N equidistant values over the support of the distribution.
        :return:
            Cumulative density values along the support of the Weibull distribution.
        """
        if points is not None:
        #    w = 1 - np.exp(-1.0 * ( (points) / (self.scale * 1.0)  )**self.shape)
        #    return w
            return self.parent.cdf(points)
        else:
            raise ValueError( 'Please digit an input for getCDF method')
