import numpy as np
from copy import deepcopy
from sklearn.metrics import mean_squared_error

from equadratures.parameter import Parameter
from equadratures.poly import Poly
from equadratures.basis import Basis

class PolyTree(object):

	def __init__(self, max_depth=5, min_samples_leaf=10, order=3):
		self.max_depth = max_depth
		self.min_samples_leaf = min_samples_leaf
		self.order = order
		self.tree = None

	def get_polys(self):
		def _search_tree(node, polys):
			if node["children"]["left"] == None and node["children"]["right"] == None:
				polys.append(node["poly"])
			
			if node["children"]["left"] != None:
				polys = _search_tree(node["children"]["left"], polys)
			
			if node["children"]["right"] != None:
				polys = _search_tree(node["children"]["right"], polys)

			return polys
		
		return _search_tree(self.tree, [])

	def fit(self, X, y):

		max_depth = self.max_depth
		min_samples_leaf = self.min_samples_leaf

		def _build_tree():

			global index_node_global

			def _create_node(X, y, depth, container):
				poly_loss, poly = _fit_poly(X, y, self.order)

				node = {"name": "node",
						"index": container["index_node_global"],
						"loss": poly_loss,
						"poly": poly,
						"data": (X, y),
						"n_samples": len(X),
						"j_feature": None,
						"threshold": None,
						"children": {"left": None, "right": None},
						"depth": depth}
				container["index_node_global"] += 1

				return node

			def _split_traverse_node(node, container, order):

				result = _splitter(node, max_depth=max_depth,
								   min_samples_leaf=min_samples_leaf,
								   order=order)
				if not result["did_split"]:
					return

				node["j_feature"] = result["j_feature"]
				node["threshold"] = result["threshold"]

				del node["data"]

				(X_left, y_left), (X_right, y_right) = result["data"]
				poly_left, poly_right = result["polys"]

				node["children"]["left"] = _create_node(X_left, y_left, node["depth"]+1, container)
				node["children"]["right"] = _create_node(X_right, y_right, node["depth"]+1, container)
				node["children"]["left"]["poly"] = poly_left
				node["children"]["right"]["poly"] = poly_right

				# Split nodes
				_split_traverse_node(node["children"]["left"], container, order)
				_split_traverse_node(node["children"]["right"], container, order)


			container = {"index_node_global": 0}
			root = _create_node(X, y, 0, container)
			_split_traverse_node(root, container, self.order)

			return root

		self.tree = _build_tree()
	
	def predict(self, X):
		assert self.tree is not None
		def _predict(node, x):
			no_children = node["children"]["left"] is None and \
						  node["children"]["right"] is None
			if no_children:
				y_pred_x = node["poly"].get_polyfit(np.array(x))[0]
				return y_pred_x
			else:
				if x[node["j_feature"]] <= node["threshold"]:  # x[j] < threshold
					return _predict(node["children"]["left"], x)
				else:  # x[j] > threshold
					return _predict(node["children"]["right"], x)
		y_pred = np.array([_predict(self.tree, np.array(x)) for x in X])
		return y_pred

	def loss(self, X, y, y_pred):
		return mean_squared_error(X, y, y_pred)

def _splitter(node, max_depth, min_samples_leaf, order):
	# Extract data
	X, y = node["data"]
	depth = node["depth"]
	N, d = X.shape

	# Find feature splits that might improve loss
	did_split = False
	loss_best = node["loss"]
	data_best = None
	polys_best = None
	j_feature_best = None
	threshold_best = None

	# Perform threshold split search only if node has not hit max depth
	if (depth >= 0) and (depth < max_depth):

		for j_feature in range(d):

			threshold_search = []
			for i in range(N):
				threshold_search.append(X[i, j_feature])

			# Perform threshold split search on j_feature
			for threshold in threshold_search:

				# Split data based on threshold
				(X_left, y_left), (X_right, y_right) = _split_data(j_feature, threshold, X, y)
				N_left, N_right = len(X_left), len(X_right)

				# Splitting conditions
				split_conditions = [N_left >= min_samples_leaf,
									N_right >= min_samples_leaf]

				# Do not attempt to split if split conditions not satisfied
				if not all(split_conditions):
					continue

				# Compute weight loss function
				loss_left, poly_left = _fit_poly(X_left, y_left, order)
				loss_right, poly_right = _fit_poly(X_right, y_right, order)
				loss_split = (N_left*loss_left + N_right*loss_right) / N

				# Update best parameters if loss is lower
				if loss_split < loss_best:
					did_split = True
					loss_best = loss_split
					polys_best = [poly_left, poly_right]
					data_best = [(X_left, y_left), (X_right, y_right)]
					j_feature_best = j_feature
					threshold_best = threshold

	# Return the best result
	result = {"did_split": did_split,
			  "loss": loss_best,
			  "polys": polys_best,
			  "data": data_best,
			  "j_feature": j_feature_best,
			  "threshold": threshold_best,
			  "N": N}

	return result

def _fit_poly(X, y, order):

	N, d = X.shape
	myParameters = []

	for d in range(d):
		d = [X[i,d] for i in range(N)]
		d_min = min(d)
		d_max = max(d)
		myParameters.append(Parameter(distribution='Uniform', lower=d_min, upper=d_max, order=order))

	myBasis = Basis('tensor-grid')
	poly = Poly(myParameters, myBasis, method='least-squares', sampling_args={'sample-points':X, 'sample-outputs':y})
	poly.set_model()

	return mean_squared_error(y, poly.get_polyfit(X)), poly

def _split_data(j_feature, threshold, X, y):
	idx_left = np.where(X[:, j_feature] <= threshold)[0]
	idx_right = np.delete(np.arange(0, len(X)), idx_left)
	assert len(idx_left) + len(idx_right) == len(X)
	return (X[idx_left], y[idx_left]), (X[idx_right], y[idx_right])


import random

def f(x1,x2):
	return x1 ** 2 - x2 ** 2

def sample():
	X, y = [], []
	for i in range(50):
		x1, x2 = random.random(), random.random()        
		X.append(np.array([x1, x2]))
		y.append(np.array(f(x1, x2)))
	return np.array(X), np.array(y)

X, y = sample()
y = np.reshape(y, (y.shape[0], 1))

tree = PolyTree()
tree.fit(X, y)

for poly in tree.get_polys():
	print(poly.get_mean_and_variance())
