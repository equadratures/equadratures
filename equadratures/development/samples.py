"""The polynomial parent class."""
#from equadratures.__stats__ import Statistics
from equadratures.parameter import Parameter
#from equadratures.__polycs__ import Polycs
#from equadratures.__polyint__ import Polyint
#from equadratures.__polyreg__ import Polyreg
from equadratures.basis import Basis
#from equadratures.__samples__ import Samples
import pickle
import numpy as np



class Samples(object):


"""
samples : {string, dict}
        The first argument to this input specifies the sampling strategy. Avaliable options are:
            - 'monte-carlo'
            - 'latin-hypercube'
            - 'induced-sampling'
            - 'christoffel-sampling'
            - 'sparse-grid'
            - 'tensor-grid'
            - 'user-defined'
        The second argument to this input is a dictionary, that naturally depends on the chosen string.
        Note that 'monte-carlo', 'latin-hypercube', 'induced-sampling' and 'christoffel-sampling' are random
        sampling techniques and thus their output will vary with each instance; initialization of a random seed
        is recommended to facilitate reproducibility. All these four techniques and 'tensor-grid' have a similar
        dict structure, that comprises of the fields:
            sampling-ratio : double
                The ratio of the number of samples to the number of coefficients (cardinality of the basis). Should
                be greater than 1.0 for 'least-squares'.
            subsampling-optimisation: str
                The type of subsampling required. In the aforementioned four sampling strategies, we generate a logarithm
                factor of samples above the required amount and prune down the samples using an optimisation technique.
                Avaliable options include:
                    - 'qr'
                    - 'lu'
                    - 'svd'
                    - 'newton'
        There is a separate dictionary structure for a 'sparse-grid'; the dictionary has the form:
            growth-rule : string
                Two growth rules are avaliable:
                    - 'linear'
                    - 'exponential'
                The growth rule specifies the relative increase in points when going from one level to another
            level : int
                The level parameter dictates the maximum degree of exactness associated with the quadrature rule.
        Finally, for the 'user-defined' scenario, there are two inputs in the dictionary:
            input : numpy ndarray
                The shape of this array will have size (observations, dimensions).
            output : numpy ndarray
                The shape of this output array will have size (observations, 1).
"""