"""Parameter v2.0."""
from numpy.random import uniform
from equadratures.distributions.template import Distribution

from equadratures.distributions.analytical import Analytical
from equadratures.distributions.beta import Beta
from equadratures.distributions.cauchy import Cauchy
from equadratures.distributions.chebyshev import Chebyshev
from equadratures.distributions.chi import Chi
from equadratures.distributions.chisquared import Chisquared
from equadratures.distributions.exponential import Exponential
from equadratures.distributions.gamma import Gamma
from equadratures.distributions.gaussian import Gaussian
from equadratures.distributions.gumbel import Gumbel
from equadratures.distributions.logistic import Logistic
from equadratures.distributions.lognormal import Lognormal
from equadratures.distributions.pareto import Pareto
from equadratures.distributions.rayleigh import Rayleigh
from equadratures.distributions.studentst import Studentst
from equadratures.distributions.triangular import Triangular
from equadratures.distributions.truncated_gaussian import TruncatedGaussian
from equadratures.distributions.uniform import Uniform
from equadratures.distributions.weibull import Weibull

class Parameter(Distribution):
    """
    The class defines a Parameter object. It is the child of Distribution.

    """
    def __new__(self, distribution='uniform', **kwargs):
        if distribution.lower() == 'analytical':
            return Analytical(**kwargs)
        if distribution.lower() == 'beta':
            return Beta(**kwargs)
        if distribution.lower() == 'cauchy':
            return Cauchy(**kwargs)
        if distribution.lower() == 'chebyshev':
            return Chebyshev(**kwargs)
        if distribution.lower() == 'chi':
            return Chi(**kwargs)
        if distribution.lower() == 'chi-squared':
            return Chisquared(**kwargs)
        if distribution.lower() == 'exponential':
            return Exponential(**kwargs)
        if distribution.lower() == 'gamma':
            return Gamma(**kwargs)
        if distribution.lower() == 'gaussian' or distribution.lower() == 'normal':
            return Gaussian(**kwargs)
        if distribution.lower() == 'gumbel':
            return Gumbel(**kwargs)
        if distribution.lower() == 'logistic':
            return Logistic(**kwargs)
        if distribution.lower() == 'lognormal':
            return Lognormal(**kwargs)
        if distribution.lower() == 'pareto':
            return Pareto(**kwargs)
        if distribution.lower() == 'rayleigh':
            return Rayleigh(**kwargs)
        if distribution.lower() == 'students-t' or distribution.lower() == 't' or distribution.lower() == 'studentt':
            return Studentst(**kwargs)
        if distribution.lower() == 'triangular':
            return Triangular(**kwargs)
        if distribution.lower() == 'truncated-gaussian' or distribution.lower() == 'normal':
            return TruncatedGaussian(**kwargs)
        if distribution.lower() == 'weibull':
            return Weibull(**kwargs)
        if distribution.lower() == 'uniform' :
            return Uniform(**kwargs)
        else:
            raise ValueError('Unknown distribution specified:', distribution.lower())