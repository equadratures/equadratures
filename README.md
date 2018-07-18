![EFFECTIVE-QUADRATURES](https://static.wixstatic.com/media/dad873_3938470ea83849db8b53716c94dd20e8~mv2.png/v1/fill/w_269,h_66,al_c,usm_0.66_1.00_0.01/dad873_3938470ea83849db8b53716c94dd20e8~mv2.png)

# Effective Quadratures
NOTE: For downloading the code, please fork the code.

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

For more examples, do checkout the notebooks here: www.effective-quadratures.org.
# Documentation
We use Sphinx for code documentation. See [Read the Docs](http://www-edc.eng.cam.ac.uk/~ps583/docs/) for more information. 

# Community guidelines
If you have contributions, questions, or feedback use either the Github repository, or contact:<br>
<br>
Pranay Seshadri <br>
*University of Cambridge* <br>
