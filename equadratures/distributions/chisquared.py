"""The Chi-squared distribution."""
from equadratures.distributions.template import Distribution
import numpy as np
from scipy.stats import chi2
RECURRENCE_PDF_SAMPLES = 50000
class Chisquared(Distribution):
    """
    The class defines a Chi-squared object. It is the child of Distribution.

    :param int dofs:
		Degrees of freedom for the chi-squared distribution.
    """
    def __init__(self, **kwargs):
        first_arg = ['dofs', 'k', 'shape_parameter_A']
        second_arg = ['order', 'orders', 'degree', 'degrees']
        third_arg = ['endpoints', 'endpoint']
        fourth_arg = ['variable']
        self.name = 'chisquared'
        self.order = 2
        self.endpoints = 'none'
        self.variable = 'parameter'
        for key, value in kwargs.items():
            if first_arg.__contains__(key):
                self.dofs = value
                self.shape_parameter_A = value
            if second_arg.__contains__(key):
                self.order = value
            if third_arg.__contains__(key):
                self.endpoints = value
            if fourth_arg.__contains__(key):
                self.variable = value

        if self.dofs is None:
            self.dofs = 1
        else:
            self.dofs = int(self.dofs)

        if not isinstance(self.dofs, int) or self.dofs < 1:
            raise ValueError('Invalid parameter in chisquared distribution: dofs must be positive integer.')

        if self.dofs == 1:
            self.bounds = np.array([1e-15, np.inf])
        else:
            self.bounds = np.array([0.0, np.inf])

        self.parent = chi2(self.dofs)
        self.mean, self.variance, self.skewness, self.kurtosis = self.parent.stats(moments='mvsk')
        self.x_range_for_pdf = np.linspace(0.0, 10.0*self.mean,RECURRENCE_PDF_SAMPLES)
        super().__init__(name=self.name, \
                        lower=self.bounds[0], \
                        upper=self.bounds[1], \
                        mean=self.mean, \
                        variance=self.variance, \
                        skewness=self.skewness, \
                        kurtosis=self.kurtosis, \
                        x_range_for_pdf=self.x_range_for_pdf, \
                        order=self.order, \
                        endpoints=self.endpoints, \
                        variable=self.variable, \
                        scipyparent=self.parent)

    def get_description(self):
        """
        A description of the Chi-squared distribution.

        :param Chisquared self:
            An instance of the Chi-squared class.
        :return:
            A string describing the Chi-squared distribution.
        """
        text = ("is a chi-squared distribution; characterised by its degrees of freedom, which here is " + \
                str(self.dofs) + ".")
        return text
