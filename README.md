# equadratures

Effective Quadratures is an open-source library for *uncertainty quantification*, *machine learning*, *optimisation*, *numerical integration* and *dimension reduction* -- all using orthogonal polynomials. It is particularly useful for models / problems where output quantities of interest are smooth and continuous; to this extent it has found widespread applications in computational engineering models (finite elements, computational fluid dynamics, etc). It is built on the latest research within these areas and has both deterministic and randomized algorithms. Effective Quadratures is actively being developed by researchers at the [University of Cambridge](https://www.cam.ac.uk), [Imperial College London](https://www.imperial.ac.uk), [Stanford University](https://www.stanford.edu), [The University of Utah](https://www.utah.edu), [The Alan Turing Institute](https://www.turing.ac.uk) and the [University of Cagliari](https://www.unica.it/unica/). Effective Quadratures is a NumFOCUS affiliated project.

**Key words associated with this code**: polynomial surrogates, polynomial chaos, polynomial variable projection, Gaussian quadrature, Clenshaw Curtis, polynomial least squares, compressed sensing, gradient-enhanced surrogates, supervised learning.

## Code

The latest version of the code is v9.0.1 *Narwhal* and was released in September 2020. 

![](https://travis-ci.org/Effective-Quadratures/Effective-Quadratures.svg?branch=master)
[![](https://coveralls.io/repos/github/Effective-Quadratures/Effective-Quadratures/badge.svg?branch=master)](https://coveralls.io/github/Effective-Quadratures/Effective-Quadratures)
[![](https://badge.fury.io/py/equadratures.svg)](https://pypi.org/project/equadratures/)
[![](https://joss.theoj.org/papers/10.21105/joss.00166/status.svg)](https://joss.theoj.org/papers/10.21105/joss.00166)
[![](https://img.shields.io/pypi/pyversions/equadratures.svg)](https://pypi.python.org/pypi/equadratures)
![](https://img.shields.io/github/stars/Effective-Quadratures/Effective-Quadratures.svg?style=flat-square&logo=github&label=Stars&logoColor=white)
![](https://static.pepy.tech/badge/equadratures/week)
[![](https://img.shields.io/discourse/status?server=https%3A%2F%2Fdiscourse.equadratures.org)](https://discourse.equadratures.org)

To download and install the code please use the python package index command:

```python
pip install equadratures
```

or if you are using python3, then

```python
pip3 install equadratures
```

Alternatively you can click either on the **Fork Code** button or **Clone**. For issues with the code, please do *raise an issue* on our Github page; do make sure to add the relevant bits of code and specifics on package version numbers. We welcome contributions and suggestions from both users and folks interested in developing the code further.

Our code is designed to require minimal dependencies; current package requirements include ``numpy``, ``scipy`` and ``matplotlib``.

## Documentation, tutorials, Discourse

Code documentation and details on the syntax can be found [here](https://www.effective-quadratures.org/docs/_documentation/).

We've recently started a Discourse forum! Check it out [here](https://discourse.equadratures.org/).

## Code objectives

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

## Get in touch

Feel free to follow us via [Twitter](https://twitter.com/EQuadratures) or email us at contact@effective-quadratures.org. 


## Community guidelines

If you have contributions, questions, or feedback use either the Github repository, or get in touch. We welcome contributions to our code. In this respect, we follow the [NumFOCUS code of conduct](https://numfocus.org/code-of-conduct). 

## Acknowledgments

This work was supported by wave 1 of The UKRI Strategic Priorities Fund under the EPSRC grant EP/T001569/1, particularly the [Digital Twins in Aeronautics](https://www.turing.ac.uk/research/research-projects/digital-twins-aeronautics) theme within that grant, and [The Alan Turing Institute](https://www.turing.ac.uk).
