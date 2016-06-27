.. Effective-Quadratures documentation master file, created by
   sphinx-quickstart on Mon Jun 27 11:13:12 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Effective-Quadratures
########################
Effective-Quadratures is a suite of tools for generating polynomials for approximation, uncertainty quantification, optimization and integration. This is a research code designed for the development and testing for techniques in the areas listed above. The documentation in these pages covers the code's main functionality and its use. All functions are coded in python, so this code is not designed for speed -- however it does

There are three principal building blocks in Effective-Quadratures. These are the parameter(s), the index set (polynomial basis) and the method. Both the parameter(s) and the index set are classes with a set of attributes while the method is simply given as a string. Below we detail all of these.


Classes
================

PolyParams() class
------------------------
All definitions in Effective-Quadratures are centered

IndexSets() class
--------------------
Talk about indexset class



PolyParent() class
--------------------
A PolyParent object takes in two or three inputs, depending on what exactly the user wants. For pseudospectral tensor and sparse grid related computations, the user does not need to input the additional index_set parameter. This is only to be used in case of a least squares type coefficient estimate, where the basis terms may be different from quadrature point evaluations.  ::

   	from effective_quadratures.PolyParentFile import PolyParent

	PolyParent(uq_parameters, method, index_sets=None)
 
Existing methods include:

1. *Sparse grid*
2. *SPAM* [#f1]_.yada
3. *Tensor grid*
4. *Least squares* (using either tensor or sparse grids)

Below are several ways this object can be called ::
	
	# Sparse grid integration rule based method (do not use) 
	PolyParent(uq_parameters, "Sparse grid", index_sets)

	# Sparse pseudospectral approximation method 
	PolyParent(uq_parameters, "SPAM", index_sets)
	
	# Tensor grid pseudospectral method 
	PolyParent(uq_parameters, "Tensor grid")

	# Least squares using a tensor grid basis with tensor grid quadrature points 
	PolyParent(uq_parameters, "Least squares") 

	# Least squares using a non-tensor grid basis with tensor grid quadrature points 
	PolyParent(uq_parameters, "Least squares", index_sets) 

For other least squares type problems, we use the Effectively Subsampled Class.

.. References


Functions
====================

Integration
----------------------
Talk about integrals

Optimization
-----------------------

ComputeStats
-----------------------
Something goes here...

.. rubric:: References

[#f1] Constantine, Paul G., Michael S. Eldred, and Eric T. Phipps. "Sparse pseudospectral approximation method." Computer Methods in Applied Mechanics and Engineering 229 (2012): 1-12.
