"""The Chebyshev / Arcsine distribution."""
import numpy as np
from distribution import Distribution
from recurrence_utils import jacobi_recurrence_coefficients

class Chebyshev(Distribution):
    """
    The class defines a Chebyshev object. It is the child of Distribution.
    
    :param double lower:
		Lower bound of the support of the Chebyshev (arcsine) distribution.
	:param double upper:
		Upper bound of the support of the Chebyshev (arcsine) distribution.
    """
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper
        self.bounds = np.array([0.0, 1.0])
        if ( self.lower is not None ) and (self.upper is not None) :
            self.mean = 0.5
        self.variance = 1.0/8.0
        self.skewness = 0.0
    
    def getDescription(self):
        """
        A description of the Chebyshev (arcsine) distribution.
            
        :param Chebyshev self:
            An instance of the Chebyshev (arcsine) class.
        :return:
            A string describing the Chebyshev (arcsine) distribution.
        """
        text = "A Chebyshev (arcsine) distribution is characterised by its lower bound, which is"+str(self.lower)+" and its upper bound, which is"+str(self.upper)+"."
        return text

    def getPDF(self, N=None, points=None):
        """
        A Chebyshev probability density function.
        
        :param Chebyshev self:
            An instance of the Chebyshev (arcsine) class.
        :param integer N:
            Number of points for defining the probability density function.
        :return:
            An array of N equidistant values over the support of the Chebyshev (arcsine) distribution.
        :return:
            Probability density values along the support of the Chebyshev (arcsine) distribution.
        """
        if N is not None:
            xreal = np.linspace(self.lower, self.upper, N)
            wreal = 1.0 / (np.pi * np.sqrt( ((xreal+0.0000001) - self.lower) * (self.upper - (xreal-0.0000001)) )  )
            return xreal, wreal
        elif points is not None:
            wreal = 1.0 / (np.pi * np.sqrt( ((points+0.0000001) - self.lower) * (self.upper - (points-0.0000001)) )  )
            return wreal
        else:
            raise(ValueError, 'Please digit an input for getPDF method')

    def getCDF(self, N=None, points=None):
        """
        A Chebyshev cumulative density function.
        
        :param Chebyshev self:
            An instance of the Chebyshev class.
        :param integer N:
            Number of points for defining the cumulative density function.
        :return:
            An array of N equidistant values over the support of the Chebyshev (arcsine) distribution.
        :return:
            Cumulative density values along the support of the Chebyshev (arcsine) distribution.
        """
        if N is not None:
            xreal = np.linspace(self.lower, self.upper, N)
            wreal = 2.0 / (np.pi) * np.arcsin( np.sqrt( (xreal - self.lower)/(self.upper - self.lower) ) )
            return xreal, wreal
        elif points is not None:
            wreal = 2.0 / (np.pi) * np.arcsin( np.sqrt( (points - self.lower)/(self.upper - self.lower) ) )
            return wreal
        else:
            raise(ValueError, 'Please digit an input for getCDF method')

    def getRecurrenceCoefficients(self, order):
        """
        Recurrence coefficients for the Chebyshev distribution.
        
        :param Chebyshev self:
            An instance of the Chebyshev class.
        :param array order:
            The order of the recurrence coefficients desired.
        :return:
            Recurrence coefficients associated with the Chebyshev distribution.
        """
        ab =  jacobi_recurrence_coefficients(-0.5, -0.5, self.lower, self.upper, order)
        return ab
