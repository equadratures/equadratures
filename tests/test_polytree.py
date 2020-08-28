from unittest import TestCase
import unittest
from equadratures import *
import numpy as np
import scipy.stats as st

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

class Test_polytree(TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(0)

    def test_gen_use(self):
        X = np.linspace(0, 1, num=100)
        y = np.concatenate((25*(X[0:50]-0.25)**2 - 1.0625, 25*(X[50:100]-0.75)**2 - 1.0625))

        X, y = unison_shuffled_copies(X,y)
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]

        X_train = np.reshape(X_train, (X_train.shape[0], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], 1))
        y_train = np.reshape(y_train, (y_train.shape[0], 1))

        tree = polytree.PolyTree()
        tree.fit(X_train, y_train)
        _, _, exhaustive_r_value, _, _ = st.linregress(y_test, tree.predict(X_test).reshape(-1))

        tree = polytree.PolyTree(search='uniform')
        tree.fit(X_train, y_train)
        _, _, uniform_r_value, _, _ = st.linregress(y_test, tree.predict(X_test).reshape(-1))
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
        
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]        

        tree = polytree.PolyTree()
        tree.fit(X_train, y_train)
        _, _, exhaustive_r_value, _, _ = st.linregress(y_test, tree.predict(X_test).reshape(-1))

        tree = polytree.PolyTree(search='uniform')
        tree.fit(X_train, y_train)
        _, _, uniform_r_value, _, _ = st.linregress(y_test, tree.predict(X_test).reshape(-1))

        self.assertTrue(uniform_r_value ** 2 > 0.9)
        self.assertTrue(exhaustive_r_value ** 2 > 0.9)

    def test_M5P_prune(self):
        X = []
        y = []
        for x1 in range(0, 10):
            for x2 in range(0, 10):
                X.append(np.array([x1/10,x2/10]))
                y.append(np.exp(-(x1/10)**2 + (x2/10)**2))
        X = np.array([X])[0]
        y = np.array(y)
        X, y = unison_shuffled_copies(X,y)
        
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]        

        tree = polytree.PolyTree(tree_type="m5p")
        tree.fit(X_train, y_train)
        tree.prune(X_test,y_test)
        _, _, pruned_r_value, _, _ = st.linregress(y_test, tree.predict(X_test).reshape(-1))

        self.assertTrue(pruned_r_value ** 2 > 0.9)

if __name__== '__main__':
    unittest.main()