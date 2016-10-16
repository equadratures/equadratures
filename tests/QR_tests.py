#!/usr/bin/env python
import effective_quadratures.qr as qr
import numpy as np
# Test the QR factorization routine!
def main():

   # Test the householder vector
   x = [4.0, 2.0, 1., -2., 4., 1.]
   x = np.mat(x)
   print x

   beta, v = qr.house(x.T)
   #print beta, v
   #A = np.random.rand(8,4)
   #Q, R = qr.qr_householder(A)
   #print Q, R
   #print '-----------'
   #print Q * R
   #print A
    # Solve the least squares problem and compare the result to 
    # what we know!
    

main()
