"""Parameter v2.0."""
from equadratures.distributions.gaussian import Gaussian
from equadratures.distributions.uniform import Uniform
from equadratures.distributions.beta import Beta
from equadratures.distributions.template import Distribution
import numpy as np
from scipy.stats import norm
from scipy.special import erf, erfinv, gamma, beta, betainc, gammainc

class Parameter(Distribution):
    """
    The class defines a Parameter object. It is the child of Distribution.

    """
    def __new__(self, distribution, **kwargs):
        if distribution.lower() == 'gaussian' or distribution.lower() == 'normal':
            return Gaussian(**kwargs)
        if distribution.lower() == 'uniform' :
            return Uniform(**kwargs)
        if distribution.lower() == 'beta':
            return Beta(**kwargs)