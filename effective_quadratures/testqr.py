#!/usr/bin/python
import numpy as np
import MatrixRoutines as matrix

"""
script for testing QR
- delete later
"""
def main():
  A = np.random.rand(5,9)
  [Q,R,P] = matrix.qrColumnPivoting_mgs(A)

  print Q
  print R
  print P

main()
