from .parameter import Parameter
from .basis import Basis
from .poly import Poly
import numpy as np

#%% Multi-layer perceptron with poly activation
class Polynet(object):
    def __init__(self, training_input, training_output, num_ridges, max_iters=1, learning_rate = 0.001,
                 W=None, coeffs=None, momentum_rate = .001, opt = 'sd', poly_deg = 2, verbose = False):
        self.training_input = training_input
        self.training_output = training_output
        self.verbose = verbose
        # network architecture params
        if isinstance(num_ridges, int):
            self.num_ridges = [num_ridges]
        else:
            self.num_ridges = num_ridges

        # num_ridges is the number of hidden units at each hidden layer. Does not count the input layer
        self.num_layers = len(self.num_ridges)

        self.dims = training_input.shape[1]
        # initialize network data structures
        max_layer_size = max(self.num_ridges)
        self.poly_array = np.empty(( self.num_layers, max_layer_size), dtype=object)
        #TODO: not hardcode poly type? Have different ridges at different nodes?
        for k in range(self.num_layers):
            for j in range(self.num_ridges[k]):
                self.poly_array[k,j] = Poly(Parameter(poly_deg, distribution='uniform', lower=-3, upper=3), Basis("total order"))
        self.poly_card = self.poly_array[0,0].basis.cardinality

        layer_sizes = [self.dims] + self.num_ridges
        if W is None:
            self.W = [np.random.randn(layer_sizes[k+1], layer_sizes[k]) for k in range(self.num_layers)]
        else:
            self.W = W
        if coeffs is None:
            self.coeffs = [np.random.randn(self.num_ridges[k], self.poly_card) for k in range(self.num_layers)]
        else:
            self.coeffs = coeffs

        self.update_coeffs()
        # Note: We will keep data for every input point in one array.
        n_points = self.training_input.shape[0]
        self.delta = []
        for k in range(self.num_layers):
            self.delta.append(np.zeros((self.num_ridges[k],n_points)))
        self.act_mat = [] # Lambda
        for k in range(self.num_layers):
            self.act_mat.append(np.zeros((self.num_ridges[k], n_points)))
        self.Z = [] # node value before activation
        for k in range(self.num_layers):
            self.Z.append(np.zeros((self.num_ridges[k],n_points)))
        self.Y = [] # After activation
        for k in range(self.num_layers):
            self.Y.append(np.zeros((self.num_ridges[k],n_points)))
        self.phi = [] # basis fn evaluations
        for k in range(self.num_layers):
            self.phi.append(np.zeros((self.num_ridges[k],n_points)))

        self.evaluate_fit(self.training_input,train=True)
        # optimization params
        self.max_iters = max_iters
        self.opt = opt
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate

    def update_coeffs(self):
        for k in range(self.num_layers):
            for j in range(self.num_ridges[k]):
                current_coeffs = self.coeffs[k][j,:]
                reshaped_coeffs = np.reshape(current_coeffs, (len(current_coeffs),1))
                self.poly_array[k,j].__setCoefficients__(reshaped_coeffs)

    def evaluate_fit(self, x, train=False):
        # evaluate nn
        # if train, update with training data and also updates Z
        Z = self.Z[:]
        Y = self.Y[:]
        W = self.W[:]
        Z[0] = np.dot(W[0], x.T)
        n_points = x.shape[0]

        for k in range(1, self.num_layers):
            Y[k-1] = np.zeros((self.num_ridges[k-1], n_points))
            for i in range(self.num_ridges[k-1]):
                Y[k-1][i] = np.squeeze(np.array(self.poly_array[k,i].evaluatePolyFit(Z[k-1][i])))
            Z[k] = np.dot(W[k], Y[k-1])

        y_final = np.zeros((self.num_ridges[self.num_layers - 1], n_points))
        for i in range(self.num_ridges[self.num_layers - 1]):
            y_final[i] = np.squeeze(np.array(self.poly_array[self.num_layers - 1, i].evaluatePolyFit
                                             (Z[self.num_layers - 1][i])))
        Y[self.num_layers - 1] = y_final
        if train:
            self.Z = Z[:]
            self.Y = Y[:]
        return np.sum(y_final, axis=0)

    def update_delta(self):
        # EBP
        delta = self.delta
        Z = self.Z
        num_ridges = self.num_ridges
        n_points = Z[0].shape[1]
        num_layers = self.num_layers
        W = self.W
        act_mat = self.act_mat
        act_mat = [np.zeros((num_ridges[k], n_points)) for k in range(num_layers)]
        for k in range(num_layers):
            for i in range(num_ridges[k]):
                act_mat[k][i] = np.squeeze(np.array(self.poly_array[k,i].evaluatePolyGradFit(Z[k][i])))

        pred_points = self.evaluate_fit(self.training_input, train=True)
        delta[-1] = (pred_points - self.training_output) * act_mat[-1]

        for k in range(self.num_layers - 2, -1, -1):
            delta[k] = act_mat[k] * np.dot(W[k+1].T, delta[k+1])
        self.delta = delta[:]
        self.act_mat = act_mat[:]

    def update_phi(self):
        # update matrix of basis function evaluations. If Z is updated for train, update this as well
        phi = self.phi
        Z = self.Z
        num_ridges = self.num_ridges
        n_points = Z[0].shape[1]
        num_layers = self.num_layers
        poly_card = self.poly_card
        for k in range(num_layers):
            phi[k] = np.zeros((num_ridges[k], poly_card, n_points))
            for i in range(num_ridges[k]):
                current_poly = self.poly_array[k,i]
                phi[k][i,:,:] = np.squeeze(current_poly.getPolynomial(Z[k][i]))
        self.phi = phi[:]


    def fit(self):
        n_data = self.training_input.shape[0]
        W_change = self.W[:]
        for t in range(self.max_iters):
            W_new = self.W[:]
            coeffs_new = self.coeffs[:]
            prev_gradWs = self.W[:]
            self.update_delta()
            self.update_phi()
            for k in range(self.num_layers):
                Wk_grads = self.gradient_Wk(k)
                alphak_grads = self.gradient_alphak(k)

                # steepest descent
                if self.opt == 'sd':
                    W_new[k] = self.W[k] - self.learning_rate * Wk_grads
                elif self.opt == 'mom':
                    # momentum
                    W_new[k] = W_new[k] - self.learning_rate * Wk_grads - self.momentum_rate * W_change[k]
                elif self.opt == 'adapt':
                    # adaptive learning rate
                    if t > 0:
                        current_loss = self.loss()
                        if prev_loss < current_loss:
                            self.learning_rate *= 0.5
                        else:
                            self.learning_rate *= 1.1
                    W_new[k] = self.W[k] - self.learning_rate * Wk_grads

                coeffs_new[k] = self.coeffs[k] - self.learning_rate * np.sum(alphak_grads,axis=2)
                W_change[k] = W_new[k] - self.W[k]

            prev_loss = self.loss()
            self.W = W_new[:]
            self.coeffs = coeffs_new[:]
            self.update_coeffs()

            if t % 100 == 0 and self.verbose:
                print('iter: %d' % t)
                print('loss: %f' % self.loss())

                print('rate: %.10f' % self.learning_rate)
            if np.isnan(self.loss()):
                print('Optimization diverged... Consider lowering learning rate.')
                break
            if self.loss() / n_data < 1e-4 and self.verbose:
                print('per point loss is < 0.0001, breaking')
                break

    def loss(self):
        return np.sum((self.evaluate_fit(self.training_input) - self.training_output)**2)

    def gradient_Wk(self, k):
        # Calculate grad of E wrt W^{(k)}, the weights of the k-th layer, summed over all training points
        if k > 0:
            return np.dot(self.delta[k],  self.Y[k-1].T)
        else:
            return np.dot(self.delta[k],  self.training_input)

    def gradient_alphak(self, k):
        # Calculate grad of E wrt alpha^{(k)} the poly coeff matrix at k-th layer, summed over all training points
        Wd = self.phi[k].copy()
        if k == self.num_layers - 1:
            Wd_one = self.evaluate_fit(self.training_input) - self.training_output
        else:
            Wd_one = np.dot(self.W[k + 1].T, self.delta[k + 1])
        for c in range(Wd.shape[1]):
            Wd[:,c,:] = Wd_one
        return self.phi[k] * Wd

    # def regress_coeffs(self, x, y):
    #     # Stack vandermonde matrices to regress new coeffs for all polys together
    #     all_V = []
    #     for i in range(self.num_ridges):
    #         V = self.poly_list[i].getPolynomial(np.dot(x, self.W[:,i]))
    #         all_V.append(V.T)
    #     stack_V = np.hstack(all_V)
    #     # print(np.linalg.cond(all_V[0]))
    #     # print(np.linalg.cond(all_V[1]))
    #     # print(np.linalg.cond(stack_V))
    #     new_coeffs = np.linalg.lstsq(stack_V, Y, rcond=None)[0]
