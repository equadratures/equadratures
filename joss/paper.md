---
title: 'Effective-Quadratures (EQ): Polynomials for Computational Engineering Studies'
tags:
  - effective quadrature subsampling
  - tensor and sparse grids
  - uncertainty quantification
  - surrogate modelling
  - polynomial interpolation and approximation
authors:
 - name: Pranay Seshadri
   orcid: 0000-0002-7351-012X
   affiliation: University of Cambridge
 - name: Geoffrey Parks
   orcid: 0000-0001-8188-5047
   affiliation: University of Cambridge
date: 23 November 2016
bibliography: paper.bib
---

# Summary

Effective-Quadratures (EQ) is a suite of tools for generating polynomials for parametric computational studies. These polynomials may be used to compute approximations to scientific models; tailored specifically for uncertainty quantification, approximation and numerical integration studies. This is a research code designed for the development and testing of new polynomial techniques in the areas listed above. EQ is written entirely in python and requires Numpy, Scipy and Matplotlib.

For a computational engineering problem, albeit analytical or a black-box model, EQ constructs a polynomial approximation (or interpolant) by sampling the model at specific points in its parameter space. This may be done using tensor or sparse grids—routines for which are available in EQ—based on the work of [@Xiu2002, @Constantine2012]. The code also has routines for effectively subsampling an existing tensor grid for computing least squares approximations, based on the work of [@Seshadri2016], which uses a QR column pivoting heuristic. Once these polynomials have been generated, they may be sampled, plotted, or used for computing moments. 

The need for this software is two fold: (1) to replicate state-of-the-art results in polynomial-based uncertainty quantification (for which the number of open source codes are limited), and (2) to provide an easy-to-use platform for carrying out further research in multivariate polynomial generation. 

# References
[@Xiu2002] Xiu, Dongbin, and George Em Karniadakis. "The Wiener--Askey polynomial chaos for stochastic differential equations." SIAM journal on scientific computing 24.2 (2002): 619-644.

[@Constantine2012] Constantine, Paul G., Michael S. Eldred, and Eric T. Phipps. "Sparse pseudospectral approximation method." Computer Methods in Applied Mechanics and Engineering 229 (2012): 1-12.

[@Seshadri2016] Seshadri, Pranay, Akil Narayan, and Sankaran Mahadevan. "Effectively Subsampled Quadratures For Least Squares Polynomial Approximations." arXiv preprint arXiv:1601.05470 (2016).
