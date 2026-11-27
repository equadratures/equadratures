"""The Student's T distribution."""
from equadratures.distributions.template import Distribution
import numpy as np
from scipy.stats import t
RECURRENCE_PDF_SAMPLES = 50000
class Studentst(Distribution):
    """
    The class defines a Studentst object. It is the child of Distribution.

    :param int dofs:
		Degrees of freedom for the Student's T distribution.
    """
    def __init__(self, **kwargs):
        first_arg = ['dofs', 'nu', 'shape_parameter_A']
        second_arg = ['order', 'orders', 'degree', 'degrees']
        third_arg = ['endpoints', 'endpoint']
        fourth_arg = ['variable']
        self.name = 'studentst'
        self.order = 2
        self.endpoints = 'none'
        self.variable = 'parameter'
        for key, value in kwargs.items():
            if first_arg.__contains__(key):
                self.dofs = value
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
            raise ValueError('Invalid parameter in studentst distribution: dofs must be positive integer.')

        self.bounds = np.array([-np.inf, np.inf])

        self.parent = t(df=self.dofs)
        self.mean, self.variance, self.skewness, self.kurtosis = self.parent.stats(moments='mvsk')
        self.x_range_for_pdf = np.linspace(-5.0, 5.0,RECURRENCE_PDF_SAMPLES)
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
        A description of the Studentst distribution.

        :param Studentst self:
            An instance of the Student's T class.
        :return:
            A string describing the Student's T distribution.
        """
        text = ("is a student's t distribution; characterised by its degrees of freedom, which here is " + \
                str(self.dofs) + ".")
        return text
