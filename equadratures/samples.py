from equadratures.induced_distributions import InducedSampling


class Sampling:

    """
    Definition of a sampling object
    A unified interface for generation
    of quadrature sampling points

    Parameters
    ----------

    parameters : list
        A list of parameters,
        each element of the list is
        an instance of the Parameter class.

    basis : Basis
        An instance of the Basis class
        corresponding to the multi-index set used.

    samples : A Tuple of (String, Dict)
        The first argument to this input specifies the sampling strategy.
        Avaliable options are:
            - 'monte-carlo'
            - 'latin-hypercube'
            - 'induced-sampling'
            - 'christoffel-sampling'
            - 'sparse-grid'
            - 'tensor-grid'
            - 'user-defined'

        The second argument to this input is a dictionary,
        that naturally depends on the chosen string.

        Note that:
            - 'monte-carlo',
            - latin-hypercube'
            - 'induced-sampling'
            - 'christoffel-sampling'
        are random sampling techniques
        thus their output will vary with each instance;
        initialization of a random seed is recommended to facilitate reproducibility.
        All these four techniques and 'tensor-grid' have a similar dict structure,
        that comprises of the fields:

            sampling-ratio : double
                The ratio of the number of samples
                to the number of coefficients (cardinality of the basis).
                Should be greater than 1.0 for 'least-squares'.

            subsampling-optimisation: String

                The type of subsampling required.
                In the aforementioned four sampling strategies,
                we generate a logarithm factor of samples above the required amount
                and prune down the samples using an optimisation technique.

                Avaliable options include:
                    - 'qr'
                    - 'lu'
                    - 'svd'
                    - 'newton'

        There is a separate dictionary structure for
            - 'sparse-grid';
        the dictionary has the form:

            growth-rule : string
                Two growth rules are avaliable:
                    - 'linear'
                    - 'exponential'
                The growth rule specifies
                the relative increase in points
                when going from one level to another

            level : int
                the maximum degree of exactness
                associated with the quadrature rule.

        For the argument
            - 'user-defined'
        There are two inputs in the dictionary:

            quadrature-points : numpy ndarray
                The shape of this array will have size:
                (number of observations, dimensions).
    """

    def __init__(self, parameters, basis, samples):

        if not isinstance(samples, tuple):
            raise TypeError("The input variable samples should be a tuple got: ",
                            type(samples))
        if not (
                len(samples) == 2
                and isinstance(samples[0], str)
                and isinstance(samples[1], dict)
                ):
            raise ValueError(
                    "The samples input should be a tuple of method string\
                    and a dictionary of additional Arguments"
                    )

        # Initialise Sampling Class
        method = samples[0]
        arguments = samples[1]

        try:
            # TODO package the other 3 methods in classes
            # and place them in the hash table below
            random_samples = {'monte-carlo': "place_holder_MC",
                              'latin-hypercube': "place_holder_LH",
                              'induced-sampling': InducedSampling,
                              'christoffel-sampling': "place_holder_Christoffel"}
            sampling_method = random_samples[method]
            sampling_ratio = arguments["sampling-ratio"]
            subsampling = arguments["subsampling-optimisation"]
            self.sampling_class = sampling_method(parameters,
                                                  basis,
                                                  sampling_ratio,
                                                  subsampling)

        # Except the method is not in the hash table of random methods
        except KeyError:

            if method == "sparse-grid":
                growth_rule = arguments["growth-rule"]
                level = arguments["level"]
                # TODO define a sparse grid sampling class
                # And insert it below :)
                self.sampling_class = (parameters, basis, growth_rule, level)

            if method == "custom":
                quadrature_points = arguments["sampling-points"]
                self.sampling_class = Custom(quadrature_points)

        def samples(self):
            """
            returns the quadrature points array
            from the designated sampling method
            Note: All sampling classes interfaced here
            should have a "samples" method which returns
            the sampling points as a numpy ndarray
            """
            quadrature_points = self.sampling_class.samples()
            return quadrature_points


class Custom:

    """
    A throughput class for customised sampling point

    Parameters:

    quadrature_points: ndarray
        A numpy array of sampling points coordinates
        The shape of this array will have size:
        (number of observations, dimensions).
    """

    def __init__(self, quadrature_points):
        self.quadrature_points = quadrature_points

    def samples(self):
        """
        returns:
        quadrature_points: ndarray
            A numpy array of sampling points coordinates
            The shape of this array will have size:
            (number of observations, dimensions).
        """
        return self.quadrature_points
