.. Effective-Quadratures documentation master file, created by
   sphinx-quickstart on Mon Jun 27 11:13:12 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Effective-Quadratures
########################
Effective-Quadratures is a suite of tools for generating polynomials for parametric computational studies. These polynomials may be used to compute approximations to scientific models or more specifically for uncertainty quantification, optimization and numerical integration. This is a research code designed for the development and testing of new polynomial techniques in the areas listed above. The documentation in these pages covers the code's main functionality and its use. All functions are coded in python, so as such this code is not designed for speed. Our motivation for using python is to make this package easily accessible to engineers and scientists, regardless of their scientific computation prowess.

There are three principal building blocks in Effective-Quadratures. These are the parameter(s), the index set (polynomial basis) and the method. Both the parameter(s) and the index set are classes with a set of attributes while the method is simply given as a string. Below we detail all of these.

Quick Start Guide
===================
1. How do I compute statistics of a model?
2. How do I integrate a function?
3. How do I get a polynomial approximation of my model?
4. How do I plot my model's coefficients?

Advanced Guide
===================
1. How do I use the compressed sensing utlities?
2. How do I use effective quadrature subsampling?
3. How do I leverage gradients from my model?


Under the hood: the classes()
===========================
Below we detail the various classes and their functionalities. Understanding the structures of these classes is key to leveraging the full range of

PolyParams() class
------------------------
All polynomials generated in Effective-Quadratures are orthonormal according to some weight. ::

    """ An uncertain parameter.
    Attributes:
        param_type: The distribution associated with the parameter
        lower_bound: Lower bound of the parameter
        upper_bound: Upper bound of the parameter
        shape_parameter_A: Value of the first shape parameter
        shape_parameter_B: Value of the second shape parameter
    """
    def __init__(self, param_type, lower_bound, upper_bound, shape_parameter_A, shape_parameter_B, derivative_flag, order):

Below we provide a few examples for how to declare such an object:


IndexSets() class
--------------------
The IndexSet() class is used to establish the basis terms involved in the polynomial computation. The class constructor requires two inputs and also takes in two additional inputs: ::

    from effective_quadratures.IndexSets import IndexSet
    # Constructor definition
    def __init__(self, index_set_type, orders, level=None, growth_rule=None):



.. sidebar:: Sidebar Title
:index_set_type: *(String)* The type of index set. Options incldue "Tensor grid", "Total order", "Hyperbolic cross" and "Sparse grid"
:order: *(List of integers)* The maximum order along each dimension
:level: *(Integer)* The level of a sparse grid
:growth_rule: *(String)* The growth strategy for a sparse grid. Options include "exponential" and "linear"

Subsequent indented lines comprise
the body of the sidebar, and are
interpreted as body elements.


PolyParent() class
--------------------
A PolyParent object takes in two or three inputs, depending on the chosen method. For pseudospectral tensor and sparse grid related computations, the user does not need to input the additional index_set parameter. This is only to be used in case of a least squares type coefficient estimate, where the basis terms may be different from quadrature point evaluations.  ::

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
We can use polynomials for integration, optimization and statistics computations.

Integration
----------------------
How do we numerically compute the integral of a function? The integration function has a few routines that may be used for numerical computation of an integral. Below we present a few sample calls ::

    import effective_quadratures.Integrals as int

    "------------Sparse grid routines------------"
    points, weights = int.sparseGrid(uqParameters, indexSet)

    "------------Tensor grid routines------------"
    # Call below will use the order associated with each uqParameter
    points, weights = int.tensorGrid(listOfParameters)
    # Call below will use the order associated with the index set
    points, weights = int.tensorGrid(listOfParameters, listOfOrders)

    "------------Scattered data sets------------"
    points, weights = int.scatteredPoints(x_data)


Optimization
-----------------------
Optimize a model by assuming it can be well represented by a polynomial surrogate. For global models with constraints we resort to basic least squares ideas with reguarlization. For local models we use a trust region strategy.

ComputeStats
-----------------------
This is the main uncertainty quantification function. In addition to computing moments, we should also provide a PDF.

.. rubric:: References

[#f1] Constantine, Paul G., Michael S. Eldred, and Eric T. Phipps. "Sparse pseudospectral approximation method." Computer Methods in Applied Mechanics and Engineering 229 (2012): 1-12.
