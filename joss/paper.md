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

For a computational engineering problem, albeit analytical or a black-box model, EQ constructs a polynomial approximation (or interpolant) by sampling the model at specific points in its parameter space. This may be done using tensor or sparse grids—routines for which are available in EQ—based on the work of [@Xiu2002, @Constantine2012]. The code also has routines for effectively subsampling an existing tensor grid for computing least squares approximations, based on the work of [@Seshadri2016], which uses a QR column pivoting heuristic. Once these polynomials have been generated, they be sampled, plotted or used for computing moments. 

# References

