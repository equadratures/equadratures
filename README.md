*************************
Effective Quadratures
*************************

Effective Quadratures is an open-source library for *uncertainty quantification*, *machine learning*, *optimisation*, *numerical integration* and *dimension reduction* -- all using orthogonal polynomials. It is particularly useful for models / problems where output quantities of interest are smooth and continuous; to this extent it has found widespread applications in computational engineering models (finite elements, computational fluid dynamics, etc). It is built on the latest research within these areas and has both deterministic and randomized algorithms. Effective Quadratures is actively being developed by researchers at the `University of Cambridge <https://www.cam.ac.uk>`__ , `Stanford University <https://www.stanford.edu>`__, `The University of Utah <https://www.utah.edu>`__, `The Alan Turing Institute <https://www.turing.ac.uk>`__ and the `University of Cagliari <https://www.unica.it/unica/>`__.  

**Key words associated with this code**: polynomial surrogates, polynomial chaos, polynomial variable projection, Gaussian quadrature, Clenshaw Curtis, polynomial least squares, compressed sensing, gradient-enhanced surrogates, supervised learning.

Code
***************

The latest version of the code is version 8.0 and was released in August 2019. 

.. image:: https://travis-ci.org/Effective-Quadratures/Effective-Quadratures.svg?branch=master
	:target: https://travis-ci.org/Effective-Quadratures/

.. image:: https://coveralls.io/repos/github/Effective-Quadratures/Effective-Quadratures/badge.svg?branch=master
	:target: https://coveralls.io/github/Effective-Quadratures/Effective-Quadratures?branch=master

.. image:: https://badge.fury.io/py/equadratures.svg
	:target: https://pypi.org/project/equadratures/

.. image:: https://joss.theoj.org/papers/10.21105/joss.00166/status.svg
	:target: https://doi.org/10.21105/joss.00166

.. image:: https://img.shields.io/pypi/pyversions/ansicolortags.svg

\

.. image:: https://img.shields.io/github/stars/Effective-Quadratures/Effective-Quadratures.svg?style=flat-square&logo=github&label=Stars&logoColor=white
	:target: https://github.com/Effective-Quadratures/Effective-Quadratures

.. image:: https://img.shields.io/pypi/dm/equadratures.svg?style=flat-square
	:target: https://pypistats.org/packages/equadratures

To download and install the code please use the python package index command:

.. code::
	
	pip install equadratures

or if you are using python3, then

.. code::
	
	pip3 install equadratures

Alternatively you can visit our `GitHub page <https://github.com/Effective-Quadratures/Effective-Quadratures>`__ and click either on the **Fork Code** button or **Clone**. For issues with the code, please do *raise an issue* on our Github page; do make sure to add the relevant bits of code and specifics on package version numbers. We welcome contributions and suggestions from both users and folks interested in developing the code further.

Our code is designed to require minimal dependencies; current package requirements include ``numpy``, ``scipy`` and ``matplotlib``.


Code objectives
***********

Specific goals of this code include:

* probability distributions and orthogonal polynomials
* supervised machine learning: regression and compressive sensing
* numerical quadrature and high-dimensional sampling
* transforms for correlated parameters
* computing moments from models and data-sets
* sensitivity analysis and Sobol' indices
* data-driven dimension reduction
* ridge approximations and neural networks
* surrogate-based design optimisation 


Papers (theory and applications)
***************************************

- Wong, C. Y., Seshadri, P., Parks, G. T., (2019) Extremum Global Sensitivity Analysis with Least Squares Polynomials and their Ridges. `Preprint <https://arxiv.org/abs/1907.08113>`__.

- Wong, C. Y., Seshadri, P., Parks, G. T., Girolami, M., (2019) Embedded Ridge Approximations: Constructing Ridge Approximations Over Localized Scalar Fields For Improved Simulation-Centric Dimension Reduction. `Preprint <https://arxiv.org/abs/1907.07037>`__.

- Seshadri, P., Iaccarino, G., Ghisu, T., (2019) Quadrature Strategies for Constructing Polynomial Approximations. *Uncertainty Modeling for Engineering Applications*. Springer, Cham, 2019. 1-25. `Paper <https://link.springer.com/chapter/10.1007/978-3-030-04870-9_1>`__. `Preprint <https://arxiv.org/pdf/1805.07296.pdf>`__.

- Seshadri, P., Narayan, A., Sankaran M., (2017) Effectively Subsampled Quadratures for Least Squares Polynomial Approximations." *SIAM/ASA Journal on Uncertainty Quantification* 5.1 : 1003-1023. `Paper <https://epubs.siam.org/doi/abs/10.1137/16M1057668>`__.

- Seshadri, P., Parks, G. T., (2017) Effective-Quadratures (EQ): Polynomials for Computational Engineering Studies, *Journal of Open Source Software*, 2(11), 166, `Paper <http://joss.theoj.org/papers/ba651f2b3608a5d2b085af06b1108747>`__.

- Kalra, T. S., Aretxabaleta, A., Seshadri, P., Ganju, N. K., Beudin, A. (2017). Sensitivity Analysis of a Coupled Hydrodynamic-Vegetation Model Using the Effectively Subsampled Quadratures Method (ESQM v5. 2). *Geoscientific Model Development*, 10(12), 4511. `Paper <https://www.repository.cam.ac.uk/bitstream/handle/1810/270655/Kalra_et_al-2017-Geoscientific_Model_Development_Discussions-VoR.pdf?sequence=4>`__.

Get in touch
***************

Feel free to follow us via `Twitter <https://twitter.com/EQuadratures>`__ or email us at contact@effective-quadratures.org. 


Community guidelines
***************
If you have contributions, questions, or feedback use either the Github repository, or contact: contact <at> effective-quadratures.org
