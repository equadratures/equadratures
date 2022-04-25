import numpy as np


class Graph(object):
    """
    Framework for an undirected graph.


    """
    def __init__(self):


    def add_node(self, node):
        # add node

    def add_edge(self, node_1, node_2):
        # add an edge to the graph

    def remove(self, item):
        # remove either an edge or a node

    def reveal(self):
        # prints out the edges and nodes

    def connectivity(self):
        # connectivity of the graph
        
    def degree_matrix(self):
        # Returns the degree matrix of the graph
        return np.diag(self.connectivity())

    def adjacency_matrix(self):
        # Returns the adjacency matrix of the graph
        A = np.zeros((self.num_nodes, self.num_nodes))
        mapping_dict = {}
        for i, edge in enumerate(self):
            mapping_dict[edge] = i
        for node in self:
            for num_nodes in node.neighbors():
                A[mapping_dict[node], mapping_dict[num_nodes]] = 1
        return A

    def laplacian_matrix(self):
        # Returns the laplacian matrix of the graph
        return self.degree_matrix() - (self.adjacency_matrix() > 0 )

    def incidence_matrix(self):
        # Returns the incidence matrix of the graph
        I = np.zeros((self.num_nodes, len(self.edges)))
        mapping_dict = {}
        for i, edge in enumerate(self.edges):
            mapping_dict[edge] = i
        for i, node in enumerate(self):
            for i, edge in enumerate(node.edges()):
                I[i, mapping_dict[edge]] = 1
        return I
