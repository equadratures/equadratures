# Effective Quadratures
**Version 8.0**

## Instructions for adding your own sampling method.
Insert your python file in the ``sampling_methods/`` folder. In the folder above, open ``quadrature.py`` and insert the relevant header and add your method to the ``Quadrature`` constructor. To compute quadrature points, your method should only require a list of ``parameters`` and the ``basis``. 
