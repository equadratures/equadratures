#!/usr/bin/python
import numpy as np
import MatrixRoutines as matrix
import scipy.linalg as sc

"""
script for testing QR
- delete later
"""
def main():
  A = np.random.rand(5,9)
  [Q,R,P] = matrix.qrColumnPivoting_mgs(A)

  # Compare with scipy
  Q2, R2, P2 = sc.qr(A,  pivoting=True)

  print Q, Q2
  print R, R2
  print P, P2

main()
