#%%
from equadratures import *
import numpy as np
from active_subspaces.domains import hit_and_run_z
import matplotlib.pyplot as plt

d = 5
n = 2

def f_0(x,w):
    return np.sin(np.dot(x, w) * np.pi)

def f_1(x,w):
    return np.exp(np.dot(x,w))

def subspace_dist(U, V):
    return np.linalg.norm(np.dot(U, U.T) - np.dot(V, V.T), ord=2)
    #
    # if len(U.shape) == 1:
    #     return np.linalg.norm(np.outer(U, U) - np.outer(V, V), ord=2)
    # else:
    #     return np.linalg.norm(np.dot(U, U.T) - np.dot(V, V.T), ord=2)

rand_norm = np.random.randn(d,n)
W,_ = np.linalg.qr(rand_norm)
poly_list = []
N = 1000
p = 7
myBasis = Basis('Total order', [p for _ in range(d)])
params = [Parameter(order=p, distribution='uniform', lower=-1, upper=1) for _ in range(d)]
X_train = np.random.uniform(-1, 1, size=(N, d))
Y_train = np.zeros((N, n))
Y_train[:,0] = np.apply_along_axis(f_0, 1, X_train, W[:,0])
Y_train[:,1] = np.apply_along_axis(f_1, 1, X_train, W[:,1])
rsq_list = []
for i in range(n):
    poly_list.append(Polyreg(params, myBasis, training_inputs=X_train, training_outputs=Y_train[:,i],
                             no_of_quad_points=0))
    rsq_list.append(poly_list[i].getfitStatistics()[1])

my_dr = dr(training_input=X_train)

R = np.eye(n)
[eigs_emb, U_emb] = my_dr.vector_AS(poly_list, R=R)
U_emb = np.real(U_emb)

print(subspace_dist(W,np.array(U_emb[:,:n])))

#%%

def harz(W1,W2,y,N):
    U = np.hstack([W1,W2])
    Z = hit_and_run_z(N, y, W1, W2)
    yz = np.vstack([np.repeat(y[:,np.newaxis], N, axis=1), Z.T])
    return np.dot(U, yz).T

N_inactive = 50
plot_coords = np.zeros((N_inactive, 2, 10))
for t in range(10):
    rand_active_coords = np.random.uniform(-1,1,2)

    new_X = harz(U_emb[:,:n], U_emb[:,n:], rand_active_coords, N_inactive)

    new_f0 = np.apply_along_axis(f_0, 1, new_X, W[:,0])
    new_f1 = np.apply_along_axis(f_1, 1, new_X, W[:,1])

    plot_coords[:,0,t] = new_f0
    plot_coords[:,1,t] = new_f1

plt.figure()
for t in range(10):
    plt.scatter(plot_coords[:, 0, t], plot_coords[:, 1, t], s=3)
plt.show()
