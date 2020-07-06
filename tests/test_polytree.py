from unittest import TestCase
import unittest
from equadratures import *
import numpy as np
import scipy.stats as st

np.random.seed(0)
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

class TestC(TestCase):

    def test_gen_use(self):
        
        X = np.linspace(0, 1, num=100)
        y = np.concatenate((25*(X[0:50]-0.25)**2 - 1.0625, 25*(X[50:100]-0.75)**2 - 1.0625))

        X, y = unison_shuffled_copies(X,y)
        x_train, x_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]

        x_train = np.reshape(x_train, (x_train.shape[0], 1))
        x_test = np.reshape(x_test, (x_test.shape[0], 1))
        y_train = np.reshape(y_train, (y_train.shape[0], 1))

        tree = polytree.PolyTree()
        tree.fit(x_train, y_train)
        _, _, exhaustive_r_value, _, _ = st.linregress(y_test, tree.predict(x_test).reshape(-1))

        tree = polytree.PolyTree(search='uniform')
        tree.fit(x_train, y_train)
        _, _, uniform_r_value, _, _ = st.linregress(y_test, tree.predict(x_test).reshape(-1))

        self.assertTrue(uniform_r_value ** 2 > 0.9)
        self.assertTrue(exhaustive_r_value ** 2 > 0.9)

    def test_high_dim_gen_use(self):
        X = []
        y = []
        for x1 in range(0, 10):
            for x2 in range(0, 10):
                X.append(np.array([x1/10,x2/10]))
                y.append(np.exp(-(x1/10)**2 + (x2/10)**2))
        X = np.array([X])[0]
        y = np.array(y)
        X, y = unison_shuffled_copies(X,y)
        
        x_train, x_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]        

        tree = polytree.PolyTree()
        tree.fit(X, y)
        _, _, exhaustive_r_value, _, _ = st.linregress(y_test, tree.predict(x_test).reshape(-1))

        tree = polytree.PolyTree(search='uniform')
        tree.fit(x_train, y_train)
        _, _, uniform_r_value, _, _ = st.linregress(y_test, tree.predict(x_test).reshape(-1))

        self.assertTrue(uniform_r_value ** 2 > 0.9)
        self.assertTrue(exhaustive_r_value ** 2 > 0.9)

if __name__== '__main__':
    unittest.main()

#ideas: multi-dimensional, switch to r^2, generate data instead, split locations and change graphviz dependencies
