"""A set of utilities for cmputing probability and cumulative density functions."""
import numpy as np
from scipy.special import erf, erfinv, gamma, beta, betainc, gammainc
import scipy.stats as stats
def iCDF_Gaussian(xx, mu, sigma_squared):
    """
    An inverse Gaussian cumulative density function.

    :param array xx:
        A numpy array of uniformly distributed samples between [0,1].
    :param double mu:
        Mean of the Gaussian distribution.
    :param doublesigma:
        Standard deviation of the Gaussian distribution.
    :return:
        Inverse CDF samples associated with the Gaussian distribution.
    """
    return mu + np.sqrt(sigma_squared) * np.sqrt(2.0) * erfinv(2.0*xx - 1.0)
def iCDF_CauchyDistribution(xx, x0, gammavalue):
    """
    An inverse Cauchy cumulative density function.

    :param array xx:
        A numpy array of uniformly distributed samples between [0,1].
    :param double x0:
        Location parameter of the Cauchy distribution.
    :param double gammavalue:
        Scale parameter associated with the Cauchy distribution.
    :return:
        Inverse CDF samples associated with the Cauchy distribution.
    """
    return x0 + gamma * np.tan(np.pi * (xx - 0.5))
def iCDF_WeibullDistribution(xx, lambda_value, k):
    """
    An inverse Weibull cumulative density function.

    :param array xx:
        A numpy array of uniformly distributed samples between [0,1].
    :param double lambda_value:
        Scale parameter of the Weibull distribution. This parameter must be greater than 0.
    :param double k:
        Shape parameter of the Weibull distribution. This parameter must be greater than 0.
    :return:
        Inverse CDF samples associated with the Weibull distribution.
    """
    return lambda_value * (-np.log(1.0 - xx))**(1.0/k)
def iCDF_ExponentialDistribution(xx, lambda_value):
    """
    An inverse exponential cumulative density function.

    :param array xx:
        A numpy array of uniformly distributed samples between [0,1].
    :param double lambda_value:
        Rate parameter of the exponential distribution. This parameter must be greater than 0.
    :return:
        Inverse CDF samples associated with the exponential distribution.
    """
    return (-np.log(1.0 - xx))/(lambda_value)
def iCDF_BetaDistribution(xx, a, b, lower, upper):
    """
    An inverse beta cumulative density function.

    :param xx:
        A numpy array of uniformly distributed samples between [0,1].
    :param double a:
        First shape parameter of the beta distribution. This value has to be greater than 0.
    :param double b:
        Second shape parameter of the beta distribution. This value has to be greater than 0.
    :param double lower:
        Lower bound of the support of the beta distribution.
    :param double upper:
        Upper bound of the support of the beta distribution.
    :return:
        Inverse CDF samples associated with the beta distribution.
    """
    yy = []
    [x, c] = CDF_BetaDistribution(1000, a, b, lower, upper)
    for k in range(0, len(xx)):
        for i in range(0, len(x)):
            if ( (xx[k]>=c[i]) and (xx[k]<=c[i+1]) ):
                value =  float( (xx[k]-c[i])/(c[i+1]-c[i])*(x[i+1]-x[i])+x[i] )
                yy.append(value)
                break
    return yy
def iCDF_GammaDistribution(xx, k, theta):
    """
    An inverse gamma cumulative density function.

    :param xx:
        A numpy array of uniformly distributed samples between [0,1].
    :param double k:
        Shape parameter of the gamma distribution. This value has to be greater than 0.
    :param double theta:
        Scale parameter of the gamma distribution. This value has to be greater than 0.
    :return:
        Inverse CDF samples associated with the gamma distribution.
    """
    yy = []
    [x, c] = CDF_GammaDistribution(1000, k, theta)
    for k in range(0, len(xx)):
        for i in range(0, len(x)):
            if ( (xx[k]>=c[i]) and (xx[k]<=c[i+1]) ):
                value =  float( (xx[k]-c[i])/(c[i+1]-c[i])*(x[i+1]-x[i])+x[i] )
                yy.append(value)
                break
    return yy
def iCDF_ChebyshevDistribution(xx, lower, upper):
    """
    An inverse Chebyshev cumulative density function.

    :param xx:
        A numpy array of uniformly distributed samples between [0,1].
    :param double lower:
        Lower bound of the support of the Chebyshev distribution.
    :param double upper:
        Upper bound of the support of the Chebyshev distribution.
    :return:
        Inverse CDF samples associated with the Chebyshev distribution.
    """
    yy = []
    [x, c] = CDF_ChebyshevDistribution(1000, lower, upper)
    for k in range(0, len(xx)):
        for i in range(0, len(x)):
            if ( (xx[k]>=c[i]) and (xx[k]<=c[i+1]) ):
                value =  float( (xx[k]-c[i])/(c[i+1]-c[i])*(x[i+1]-x[i])+x[i] )
                yy.append(value)
                break
    return yy
def iCDF_TruncatedGaussianDistribution(xx, mu, sigma, a, b):
    """
    An inverse truncated Gaussian cumulative density function.

    :param xx:
        A numpy array of uniformly distributed samples between [0,1].
    :param double mu:
        Mean of the Gaussian distribution.
    :param doublesigma:
        Standard deviation of the Gaussian distribution.
    :param double lower:
        Lower bound of the support of the truncated Gaussian distribution.
    :param double upper:
        Upper bound of the support of the truncated Gaussian distribution.
    :return:
        Inverse CDF samples associated with the truncated Gaussian distribution.
    """
    yy = iCDF_Gaussian(xx, mu, sigma)
    for i in range(0, len(xx)):
        if(yy[i,0] < a or yy[i,0] > b):
            yy[i,0] = 0
    return yy
def iCDF_CustomDistribution(xx, data):
    """
    An inverse custom cumulative density function. Here the custom distriution is based on the input data.

    :param xx:
        A numpy array of uniformly distributed samples between [0,1].
    :param array data:
        An array of data samples from which a custom distribution is generated.
    :return:
        Inverse CDF samples associated with the custom distribution.
    """
    [x, y] = PDF_CustomDistribution(1000, data)
    c = []
    yy = []
    c.append(0.0)
    for i in range(1, len(x)):
        c.append(c[i-1]+(x[i]-x[i-1])*(y[i]+y[i-1])*.5)
    for i in range(1, len(x)):
        c[i]=c[i]/c[len(x)-1]

    for k in range(0, len(xx)):
        for i in range(0, len(x)):
            if ( (xx[k]>=c[i]) and (xx[k]<=c[i+1]) ):
                value =  float( (xx[k]-c[i])/(c[i+1]-c[i])*(x[i+1]-x[i])+x[i] )
                yy.append(value)
                break
    return yy
def CDF_GaussianDistribution(N, mu, sigma):
    """
    A Gaussian cumulative density function.

    :param integer N:
        Number of points for defining the cumulative density function.
    :param double mu:
        Mean of the Gaussian distribution.
    :param sigma:
        Standard deviation of the Gaussian distribution.
    :return:
        An array of N equidistant values over the support of the distribution.
    :return:
        Cumulative density values along the support of the distribution.
    """
    x = np.linspace(-15*sigma, 15*sigma, N)
    x = x + mu # scaling it by the mean!
    w = 0.5*(1 + erf((x - mu)/(sigma * np.sqrt(2) ) ) )
    return x, w
def CDF_TruncatedGaussianDistribution(N, mu, sigma_squared, a, b):
    """
    A truncated Gaussian cumulative density function.

    :param integer N:
        Number of points for defining the cumulative density function.
    :param double mu:
        Mean of the truncated Gaussian distribution.
    :param doublesigma:
        Standard deviation of the truncated Gaussian distribution.
    :param double lower:
        Lower bound of the support of the truncated Gaussian distribution.
    :param double upper:
        Upper bound of the support of the truncated Gaussian distribution.
    :return:
        An array of N equidistant values over the support of the distribution.
    :return:
        Cumulative density values along the support of the truncated Gaussian distribution.
    """
    def cumulative(x):
        return 0.5 * (1 + erf(x/np.sqrt(2)))
    x = np.linspace(a, b, N)
    zeta = (x - mu)/( np.sqrt(sigma_squared) )
    alpha = (a - mu)/( np.sqrt(sigma_squared) )
    beta = (b - mu)/( np.sqrt(sigma_squared) )
    Z = cumulative(beta) - cumulative(alpha)
    w = (cumulative(zeta) - cumulative(alpha))/(Z)
    return x, w
def CDF_BetaDistribution(N, a, b, lower, upper):
    """
    A beta cumulative density function.

    :param integer N:
        Number of points for defining the cumulative density function.
    :param double a:
        First shape parameter of the beta distribution. This value has to be greater than 0.
    :param double b:
        Second shape parameter of the beta distribution. This value has to be greater than 0.
    :param double lower:
        Lower bound of the support of the beta distribution.
    :param double upper:
        Upper bound of the support of the beta distribution.
    :return:
        An array of N equidistant values over the support of the distribution.
    :return:
        Cumulative density values along the support of the beta distribution.
    """
    xreal = np.linspace(lower, upper, N)
    x = np.linspace(0, 1, N)
    w = np.zeros((N,1))
    for i in range(0, N):
        w[i] = betainc(a, b, x[i])
    return xreal, w
def CDF_WeibullDistribution(N, lambda_value, k):
    """
    A Weibull cumulative density function.

    :param integer N:
        Number of points for defining the cumulative density function.
    :param double lambda_value:
        Scale parameter of the Weibull distribution. This parameter must be greater than 0.
    :param double k:
        Shape parameter of the Weibull distribution. This parameter must be greater than 0.
    :return:
        An array of N equidistant values over the support of the distribution.
    :return:
        Cumulative density values along the support of the Weibull distribution.
    """
    x = np.linspace(0, 15/k, N)
    w = 1 - np.exp(-1.0 * ( (x) / (lambda_value * 1.0)  )**k )
    return x, w
def CDF_UniformDistribution(N, lower, upper):
    """
    A uniform cumulative density function.

    :param integer N:
        Number of points for defining the cumulative density function.
    :param double lower:
        Lower bound of the support of the uniform distribution.
    :param double upper:
        Upper bound of the support of the uniform distribution.
    :return:
        An array of N equidistant values over the support of the distribution.
    :return:
        Cumulative density values along the support of the uniform distribution.
    """
    x = np.linspace(lower, upper, N)
    w = np.zeros((N, 1))
    for i in range(0, N):
        w[i] = (x[i] - lower)/(upper - lower)
    return x, w
def CDF_GammaDistribution(N, k, theta):
    """
    A gamma cumulative density function.

    :param integer N:
        Number of points for defining the cumulative density function.
    :param double k:
        Shape parameter of the gamma distribution. This value has to be greater than 0.
    :param double theta:
        Scale parameter of the gamma distribution. This value has to be greater than 0.
    :return:
        An array of N equidistant values over the support of the distribution.
    :return:
        Cumulative density values along the support of the Gamma distribution.
    """
    x = np.linspace(0, k*theta*10, N)
    w = 1.0/(gamma(k)) * gammainc(k, x/theta)
    return x, w
def CDF_CauchyDistribution(N, x0, gammavalue):
    """
    A Cauchy cumulative density function.

    :param integer N:
        Number of points for defining the cumulative density function.
    :param double x0:
        Location parameter of the Cauchy distribution.
    :param double gammavalue:
        Scale parameter associated with the Cauchy distribution.
    :return:
        An array of N equidistant values over the support of the distribution.
    :return:
        Cumulative density values along the support of the Cauchy distribution.
    """
    x = np.linspace(-15*gammavalue, 15*gammavalue, N)
    x = x + x0
    w = 1.0/np.pi * np.arctan((x - x0) / gammavalue) + 0.5
    return x, w
def CDF_ExponentialDistribution(N, lambda_value):
    """
    An exponential cumulative density function.

    :param integer N:
        Number of points for defining the cumulative density function.
    :param double lambda_value:
        Rate parameter of the exponential distribution. This parameter must be greater than 0.
    :return:
        An array of N equidistant values over the support of the distribution.
    :return:
        Cumulative density values along the support of the exponential distribution.
    """
    x = np.linspace(0, 20*lambda_value, N)
    w = 1 - np.exp(-lambda_value * x)
    return x, w
def CDF_ChebyshevDistribution(N, lower, upper):
    """
    A Chebyshev cumulative density function.

    :param integer N:
        Number of points for defining the cumulative density function.
    :param double lower:
        Lower bound of the Chebyshev distribution.
    :param double upper:
        Upper bound of the Chebyshev distribution.
    :return:
        An array of N equidistant values over the support of the distribution.
    :return:
        Cumulative density values along the support of the Chebyshev (arcsine) distribution.
    """
    xreal = np.linspace(lower, upper, N)
    wreal = 2.0 / (np.pi) * np.arcsin( np.sqrt( (xreal - lower)/(upper - lower) ) ) 
    return xreal, wreal
def CDF_CustomDistribution(N, data):
    """
    A cumulative density function associated with a given data set.

    :param integer N:
        Number of points for defining the cumulative density function.
    :param array data:
        An array of data samples from which a custom distribution is generated.
    :return:
        An array of N equidistant values over the support of the distribution.
    :return:
        Cumulative density values along the support of the custom distribution.
    """
    x, y = PDF_CustomDistribution(N, data)
    c = []
    c.append(0.0)
    for i in range(1, len(x)):
        c.append(c[i-1]+(x[i]-x[i-1])*(y[i]+y[i-1])*.5)
    for i in range(1, len(x)):
        c[i]=c[i]/c[len(x)-1]
    return x, c
def PDF_CustomDistribution(N, data):
    """
    A probability density function associated with a given data set.

    :param integer N:
        Number of points for defining the probability density function.
    :param array data:
        An array of data samples from which a custom distribution is generated.
    :return:
        An array of N equidistant values over the support of the distribution.
    :return:
        Probability density values along the support of the custom distribution.

    **Notes:**

    To obtain a probability density function from finite samples, this function uses kernel density estimation (with a Gaussian kernel).
    """
    mean = np.mean(data)
    std = np.std(data)
    lower = mean - (np.sqrt(std) * 5.0)
    upper = mean + (np.sqrt(std) * 5.0)
    xo = np.linspace(lower, upper, N)
    kernel = stats.gaussian_kde(data)
    wts = kernel(xo)
    return xo, wts
def PDF_UniformDistribution(N, lower, upper):
    """
    A uniform probability distribution.

    :param integer N:
        Number of points for defining the probability density function.
    :param double lower:
        Lower bound of the support of the uniform distribution.
    :param double upper:
        Upper bound of the support of the uniform distribution.
    :return:
        An array of N equidistant values over the support of the distribution.
    :return:
        Probability density values along the support of the uniform distribution.
    """
    x = np.linspace(lower, upper, N)
    w = 0*x + (1.0)/(upper - lower)
    return x, w
def PDF_BetaDistribution(N, a, b, lower, upper):
    """
    A beta probability distribution.

    :param integer N:
        Number of points for defining the probability density function.
    :param double a:
        First shape parameter of the beta distribution. This value has to be greater than 0.
    :param double b:
        Second shape parameter of the beta distribution. This value has to be greater than 0.
    :param double lower:
        Lower bound of the support of the uniform distribution.
    :param double upper:
        Upper bound of the support of the uniform distribution.
    :return:
        An array of N equidistant values over the support of the distribution.
    :return:
        Probability density values along the support of the beta distribution.
    """
    x = np.linspace(0, 1, N)
    w = (x**(a - 1) * (1 - x)**(b - 1))/(beta(a, b) )
    xreal = np.linspace(lower, upper, N)
    wreal = w * (1.0)/(upper - lower)
    return xreal, wreal
def PDF_GaussianDistribution(N, mu, sigma_squared):
    """
    A Gaussian probability distribution.

    :param integer N:
        Number of points for defining the probability density function.
    :param double mu:
        Mean of the Gaussian distribution.
    :param sigma:
        Standard deviation of the Gaussian distribution.
    :return:
        An array of N equidistant values over the support of the distribution.
    :return:
        Probability density values along the support of the Gaussian distribution.
    """
    x = np.linspace(-15*sigma_squared, 15*sigma_squared, N)
    x = x + mu # scaling it by the mean!
    w = 1.0/( np.sqrt(2 * sigma_squared * np.pi) ) * np.exp(-(x - mu)**2 * 1.0/(2 * sigma_squared) )
    return x, w
def PDF_WeibullDistribution(N, lambda_value, k):
    """
    A Weibull probability density function.

    :param integer N:
        Number of points for defining the probability density function.
    :param double lambda_value:
        Scale parameter of the Weibull distribution. This parameter must be greater than 0.
    :param double k:
        Shape parameter of the Weibull distribution. This parameter must be greater than 0.
    :return:
        An array of N equidistant values over the support of the distribution.
    :return:
        Probability density values along the support of the Weibull distribution.
    """
    x = np.linspace(0, 15/k, N)
    w = k/lambda_value * (x/lambda_value)**(k-1) * np.exp(-1.0 * (x/lambda_value)**k )
    return x, w
def PDF_GammaDistribution(N, k, theta):
    """
    A gamma probability density function.

    :param integer N:
        Number of points for defining the probability density function.
    :param double k:
        Shape parameter of the gamma distribution. This value has to be greater than 0.
    :param double theta:
        Scale parameter of the gamma distribution. This value has to be greater than 0.
    :return:
        An array of N equidistant values over the support of the distribution.
    :return:
        Probability density values along the support of the Gamma distribution.
    """
    x = np.linspace(0, k*theta*10, N)
    w = 1.0/(gamma(k) * theta**k) * x**(k-1) * np.exp(-x/theta)
    return x, w
def PDF_CauchyDistribution(N, x0, gammavalue):
    """
    A Cauchy probability density function.

    :param integer N:
        Number of points for defining the probability density function.
    :param double x0:
        Location parameter of the Cauchy distribution.
    :param double gammavalue:
        Scale parameter associated with the Cauchy distribution.
    :return:
        An array of N equidistant values over the support of the distribution.
    :return:
        Probability density values along the support of the Cauchy distribution.
    """
    x = np.linspace(-15*gammavalue, 15*gammavalue, N)
    x = x + x0
    w = 1.0/(np.pi * gammavalue * (1 + ((x - x0)/(gammavalue))**2) )
    return x, w
def PDF_ExponentialDistribution(N, lambda_value):
    """
    An exponential probability density function.

    :param integer N:
        Number of points for defining the probability density function.
    :param double lambda_value:
        Rate parameter of the exponential distribution. This parameter must be greater than 0.
    :return:
        An array of N equidistant values over the support of the distribution.
    :return:
        Probability density values along the support of the exponential distribution.
    """
    x = np.linspace(0, 20*lambda_value, N)
    w = lambda_value * np.exp(-lambda_value * x)
    return x, w
def PDF_ChebyshevDistribution(N, lower, upper):
    """
    A Chebyshev probability density function.

    :param integer N:
        Number of points for defining the probability density function.
    :param double lower:
        Lower bound of the Chebyshev distribution.
    :param double upper:
        Upper bound of the Chebyshev distribution.
    :return:
        An array of N equidistant values over the support of the distribution.
    :return:
        Probability density values along the support of the Chebyshev (arcsine) distribution.
    """
    xreal = np.linspace(lower, upper, N)
    wreal = 1.0 / (np.pi * np.sqrt( (xreal - lower) * (upper - xreal) )  )
    return xreal, wreal
def PDF_TruncatedGaussianDistribution(N, mu, sigma_squared, a, b):
    """
    A truncated Gaussian probability density function.

    :param integer N:
        Number of points for defining the probability density function.
    :param double mu:
        Mean of the truncated Gaussian distribution.
    :param doublesigma:
        Standard deviation of the truncated Gaussian distribution.
    :param double lower:
        Lower bound of the support of the truncated Gaussian distribution.
    :param double upper:
        Upper bound of the support of the truncated Gaussian distribution.
    :return:
        An array of N equidistant values over the support of the distribution.
    :return:
        Probability density values along the support of the truncated Gaussian distribution.
    """
    x = np.linspace(a, b, N)
    w = 1.0/( np.sqrt(2 * sigma_squared * np.pi)) * np.exp(-(x - mu)**2 * 1.0/(2 * sigma_squared) )
    w = 1.0/( np.sqrt(sigma_squared) ) * w
    first_term = GaussianCDF(b, mu,  np.sqrt(sigma_squared) )
    second_term = GaussianCDF(a, mu,  np.sqrt(sigma_squared) )
    w = w / (first_term - second_term)
    return x, w
def GaussianCDF(constant, mu, sigma_squared):
    w = 1.0/2 * (1 + erf((constant - mu)/( np.sqrt(sigma_squared) * np.sqrt(2))) )
    return w
