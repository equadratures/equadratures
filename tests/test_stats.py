from unittest import TestCase
import unittest
from equadratures import *
import numpy as np
from scipy import stats as s

class TestStats(TestCase):
    def fun1(x):
        return x[0]    
    
    def fun2(x):
        return x[0]+x[1]+x[2]
    
    def fun3(x):
        return np.exp(x[0]+x[1])
        
    self.degree = 5
    points_used = degree + 1
    self.x1 = Parameter(param_type="Uniform", lower=0.0, upper=1.0, points=points_used)
    self.x2 = Parameter(param_type="Uniform", lower=0.0, upper=1.0, points=points_used)
    self.x3 = Parameter(param_type="Uniform", lower=-1, upper=1, points=points_used)
        
    def test_1(self):
        x1 = self.x1
        degree = self.degree
        parameters = [x1]
        basis = IndexSet('Tensor grid',[degree,degree])
        uqProblem = Polyint(parameters)
        coefficients, indices, pts = uqProblem.getPolynomialCoefficients(fun1)
        stats = Statistics(coefficients, basis, parameters)
        
        x1_samples = x1.getSamples(1000000)
        f = np.zeros((1000000,1))
        
        for i in range(1000000):
            f[i,0] = fun1([x1_samples[i,0]])
        
        MC_mean = np.mean(f)
        MC_var = np.var(f)
        MC_skew = s.skew(f)
        MC_kurt = s.kurtosis(f, fisher = False)
        epsilon = 1e-5
        assert(abs((stats.mean - MC_mean)/(MC_mean + epsilon)) < 0.1)
        assert(abs((stats.variance - MC_var)/(MC_var+ epsilon)) < 0.1)
        assert(abs((stats.skewness - MC_skew)/(MC_skew+ epsilon)) < 0.1)
        assert(abs((stats.kurtosis - MC_kurt)/(MC_kurt+ epsilon)) < 0.1)


    def test_2(self):
        x1 = self.x1
        x2 = self.x2
        x3 = self.x3
        degree = self.degree
        parameters = [x1,x2,x3]
        basis = IndexSet('Tensor grid',[degree,degree])
        uqProblem = Polyint(parameters)
        coefficients, indices, pts = uqProblem.getPolynomialCoefficients(fun2)
        stats = Statistics(coefficients, basis, parameters)
        fosi = stats.getSobol(1)
        
        x1_samples = x1.getSamples(1000000)
        x2_samples = x2.getSamples(1000000)
        x3_samples = x3.getSamples(1000000)
        f = np.zeros((1000000,1))
        
        for i in range(1000000):
            f[i,0] = fun1([x1_samples[i,0], x2_samples[i,0], x3_samples[i,0]])
        
        MC_mean = np.mean(f)
        MC_var = np.var(f)
        MC_skew = s.skew(f)
        MC_kurt = s.kurtosis(f, fisher = False)
        epsilon = 1e-5
        assert(abs((stats.mean - MC_mean)/(MC_mean + epsilon)) < 0.1)
        assert(abs((stats.variance - MC_var)/(MC_var+ epsilon)) < 0.1)
        assert(abs((stats.skewness - MC_skew)/(MC_skew+ epsilon)) < 0.1)
        assert(abs((stats.kurtosis - MC_kurt)/(MC_kurt+ epsilon)) < 0.1)

    def test_3(self):
        x1 = self.x1
        x2 = self.x2
        degree = self.degree
        parameters = [x1,x2]
        basis = IndexSet('Tensor grid',[degree,degree])
        uqProblem = Polyint(parameters)
        coefficients, indices, pts = uqProblem.getPolynomialCoefficients(fun3)
        stats = Statistics(coefficients, basis, parameters)
        fosi = stats.getSobol(1)
        
        x1_samples = x1.getSamples(1000000)
        x2_samples = x2.getSamples(1000000)
        f = np.zeros((1000000,1))
        
        for i in range(1000000):
            f[i,0] = fun1([x1_samples[i,0], x2_samples[i,0]])
        
        MC_mean = np.mean(f)
        MC_var = np.var(f)
        MC_skew = s.skew(f)
        MC_kurt = s.kurtosis(f, fisher = False)
        epsilon = 1e-5
        assert(abs((stats.mean - MC_mean)/(MC_mean + epsilon)) < 0.1)
        assert(abs((stats.variance - MC_var)/(MC_var+ epsilon)) < 0.1)
        assert(abs((stats.skewness - MC_skew)/(MC_skew+ epsilon)) < 0.1)
        assert(abs((stats.kurtosis - MC_kurt)/(MC_kurt+ epsilon)) < 0.1)

if __name__ == '__main__':
    unittest.main()
