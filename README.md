![EFFECTIVE-QUADRATURES](https://static.wixstatic.com/media/dad873_3938470ea83849db8b53716c94dd20e8~mv2.png/v1/fill/w_269,h_66,al_c,usm_0.66_1.00_0.01/dad873_3938470ea83849db8b53716c94dd20e8~mv2.png)

# Effective Quadratures
[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/Effective-Quadratures/Effective-Quadratures/main/LICENSE.rst)

**What is Effective Quadratures?**
Effective Quadratures is a suite of tools for generating polynomials for approximation, uncertainty quantification (UQ), optimization and integration.  

**Why do we need it?**
 * To replicate and compare state-of-the-art techniques and results in UQ;
 * To facilitate easy programming of new techniques at a research level;
 * To have a unified, easy-to-use platform with interactive demos---tailored for industrial use.

**So, what's new in Effective Quadratures?**
 * New sampling & integration routines for more efficienct use of your model evaluations;
 * Techniques that leverage adjoint (gradient) information;
 * Plotting subroutines for quick, one-command plots.

For further details, notebooks and papers see:
<br>
www.effective-quadratures.org
<br>

# Installation
For installation on Mac and Linux systems, simply type the following into the terminal. 
```bash
> sudo python setup.py install
```
For installation on Windows, you will need [Anaconda](https://www.continuum.io/downloads#windows); select the Python 2.7 option. Upon successful installation, open the Sypder IDE and go into the Effective-Quadratures-master directory and type the following in the command window
```bash
> python setup.py install
```
This should build the code. Just make sure you include the location of effective_quadratures folder to your python file and you should be good to go. To run this code you will require python 2.7, numpy, scipy and matplotlib. 

# Simple example of use
Below we provide two simple examples that showcase some of the functionality in Effective Quadratures. In the first example we demonstrate how to construct a bi-variate quadrature rule (using a tensor grid) from two different distriutions. 
```python
from equadratures import *

x = Parameter(param_type='Gaussian', shape_parameter_A=3.0, shape_parameter_B=2.0, points=6)
y = Parameter(param_type='Weibull', shape_parameter_A=1.0, shape_parameter_B=2.2, points=4)

p = Polyint([x,y])
points, weights = p.getPointsAndWeights()
print points
[[-1.70120995  0.28565256]
 [-1.70120995  0.79778656]
 [-1.70120995  1.44042885]
 [-1.70120995  2.21498268]
 [ 0.32830185  0.28565256]
 [ 0.32830185  0.79778656]
 [ 0.32830185  1.44042885]
 [ 0.32830185  2.21498268]
 [ 2.12784518  0.28565256]
 [ 2.12784518  0.79778656]
 [ 2.12784518  1.44042885]
 [ 2.12784518  2.21498268]
 [ 3.87215482  0.28565256]
 [ 3.87215482  0.79778656]
 [ 3.87215482  1.44042885]
 [ 3.87215482  2.21498268]
 [ 5.67169815  0.28565256]
 [ 5.67169815  0.79778656]
 [ 5.67169815  1.44042885]
 [ 5.67169815  2.21498268]
 [ 7.70120995  0.28565256]
 [ 7.70120995  0.79778656]
 [ 7.70120995  1.44042885]
 [ 7.70120995  2.21498268]]
```
In the second example, we demonstrate how to approximate a complex model using a polynomial approximant. Consider a 

# Documentation
We use Sphinx for code documentation. See [Read the Docs](http://www-edc.eng.cam.ac.uk/~ps583/docs/) for more information. Additionally do check out the python notebooks and links in www.effective-quadratures.org

# Contact
For details and queries please contact:<br>
<br>
Pranay Seshadri <br>
*University of Cambridge* <br>

# Funding
This tool has been supported by funding from the Air Force Office of Scientific Research (AFOSR) under grant number FA9550-15-1-0018 and from the Engineering Physical Sciences Research Council (EPSRC) U.K.
