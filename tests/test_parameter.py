from unittest import TestCase
import unittest
from equadratures import *
import numpy as np

class TestParameter(TestCase):

    def test_param(self):
        p=Parameter(lower=-1., upper=1.)
        

if __name__== '__main__':
    unittest.main()