import numpy as np
from copy import deepcopy
from equadratures.parameter import Parameter
from equadratures.poly import Poly
from equadratures.basis import Basis
from urllib.parse import quote

class PolyTree(object):

	def __init__(self, max_depth=5, min_samples_leaf=10, order=3, basis='tensor-grid'):
		self.max_depth = max_depth
		self.min_samples_leaf = min_samples_leaf
		self.order = order
		self.basis = basis
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
				poly_loss, poly = _fit_poly(X, y, self.order, self.basis)

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

			def _split_traverse_node(node, container):

				result = _splitter(node, max_depth=max_depth,
								   min_samples_leaf=min_samples_leaf,
								   order=self.order,
								   basis=self.basis)
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
				_split_traverse_node(node["children"]["left"], container)
				_split_traverse_node(node["children"]["right"], container)


			container = {"index_node_global": 0}
			root = _create_node(X, y, 0, container)
			_split_traverse_node(root, container)

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

	def loss(self, y, y_pred):
		mse = sum([(y[i]-y_pred[i]) ** 2 for i in range(len(y))]) / len(y)
		return mse

	def get_graphviz(self, feature_names):
			
		from graphviz import Digraph
		g = Digraph('g', node_attr={'shape': 'record', 'height': '.1'})

		def build_graphviz_recurse(node, parent_node_index=0, parent_depth=0, edge_label=""):

			# Empty node
			if node is None:
				return

			# Create node
			node_index = node["index"]
			if node["children"]["left"] is None and node["children"]["right"] is None:
				threshold_str = ""
			else:
				threshold_str = "{} <= {:.1f}\\n".format(feature_names[node['j_feature']], node["threshold"])
			
			indices = []
			for i in range(len(feature_names)):
				indices.append("{} : {}\\n".format(feature_names[i], node["poly"].get_sobol_indices(1)[i,]))
			label_str = "{} n_samples = {}\\n loss = {:.6f}\\n sobol indices: {}".format(threshold_str, node["n_samples"], node["loss"], [i for i in indices])

			# Create node
			nodeshape = "rectangle"
			bordercolor = "black"
			fillcolor = "white"
			fontcolor = "black"
			g.attr('node', label=label_str, shape=nodeshape)
			g.node('node{}'.format(node_index),
				   color=bordercolor, style="filled",
				   fillcolor=fillcolor, fontcolor=fontcolor)

			# Create edge
			if parent_depth > 0:
				g.edge('node{}'.format(parent_node_index),
					   'node{}'.format(node_index), label=edge_label)

			# Traverse child or append leaf value
			build_graphviz_recurse(node["children"]["left"],
								   parent_node_index=node_index,
								   parent_depth=parent_depth + 1,
								   edge_label="")
			build_graphviz_recurse(node["children"]["right"],
								   parent_node_index=node_index,
								   parent_depth=parent_depth + 1,
								   edge_label="")

		# Build graph
		build_graphviz_recurse(self.tree,
							   parent_node_index=0,
							   parent_depth=0,
							   edge_label="")

		print('https://dreampuf.github.io/GraphvizOnline/#' + quote(str(g.source)))

def _splitter(node, max_depth, min_samples_leaf, order, basis):
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
				loss_left, poly_left = _fit_poly(X_left, y_left, order, basis)
				loss_right, poly_right = _fit_poly(X_right, y_right, order, basis)
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

def _fit_poly(X, y, order, basis):

	N, d = X.shape
	myParameters = []

	for d in range(d):
		d = [X[i,d] for i in range(N)]
		d_min = min(d)
		d_max = max(d)
		myParameters.append(Parameter(distribution='Uniform', lower=d_min, upper=d_max, order=order))

	myBasis = Basis(basis)
	try:	
		poly = Poly(myParameters, myBasis, method='least-squares', sampling_args={'mesh': 'user-defined', 'sample-points':X, 'sample-outputs':y})
		poly.set_model()
	except Exception as e:
		print(e)
		print(X,y)
	mse = float(sum([(y[i]-poly.get_polyfit(X)[i]) ** 2 for i in range(N)])) / N
	print(mse)
	return mse, poly

def _split_data(j_feature, threshold, X, y):
	idx_left = np.where(X[:, j_feature] <= threshold)[0]
	idx_right = np.delete(np.arange(0, len(X)), idx_left)
	assert len(idx_left) + len(idx_right) == len(X)
	return (X[idx_left], y[idx_left]), (X[idx_right], y[idx_right])
