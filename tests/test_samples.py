from unittest import TestCase
from equadratures.samples import Sampling
from equadratures.induced_distributions import InducedSampling
from equadratures.parameter import Parameter
from equadratures.basis import Basis


class TestSamplingGeneration(TestCase):

    def test_generate_sampling_class(self):
        # test if the method returns a function object for sampling interface
        parameters = [Parameter(1, "gaussian")]*3
        basis = Basis("total order")
        generator_class = Sampling(parameters, basis,
                                   ('induced-sampling',
                                    {"sampling-ratio": 2,
                                     "subsampling-optimisation": 'qr'}
                                    )
                                   )
        assert type(generator_class.sampling_class) == InducedSampling
