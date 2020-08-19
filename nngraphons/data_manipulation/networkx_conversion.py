import numpy as np
import networkx as nx


def np_to_nx(G):
    """Converts numpy representation of graph to networkx"""
    nx_G = nx.Graph()
    nx_G.add_nodes_from(G[0])
    nx_G.add_edges_from(G[1])
    return nx_G


def nx_to_np(G):
    """Converts networkx representation of graph to numpy arrays"""
    V = np.array(G.nodes).astype(int)
    E = nx.convert_matrix.to_pandas_edgelist(G).values.astype(int)
    return V, E
