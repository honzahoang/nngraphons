from itertools import product
import random
import numpy as np
import networkx as nx


def graphon_grow_unit_attach(x, y):
    """Growing uniform attachment graphon"""
    return 1 - np.maximum(x, y)


def graphon_inv_grow_unit_attach(x, y):
    """Inverse growing uniform attachment graphon"""
    return np.maximum(x, y)


def graphon_constant(x, y):
    """Constant 0.5 graphon"""
    return np.full(len(x), 0.5)


def graphon_complete_bipartite(x, y):
    """Complete bipartite graphon"""
    return (
        ((x <= 0.5) & (y >= 0.5))
        | ((x > 0.5) & (y < 0.5))
    ).astype(float)


def graphon_big_clique(x, y):
    """Big clique graphon"""
    return ((x <= 0.5) & (y <= 0.5)).astype(float)


def sample_graphon(W, size=None, max_size=None):
    """Returns a random graph sampled from graphon W."""
    # Number of vertices
    n_V = random.randint(2, max_size) if size is None else size
    # Randomly select nodes in graphon
    V = np.random.uniform(size=n_V)
    # All possible edges
    edge_indices = np.transpose(np.vstack(np.tril_indices(len(V))))
    edges = V[edge_indices]
    # Edge probabilites (graphon values)
    edge_probs = W(edges[:, 0], edges[:, 1])

    # Convert float values of vertices to integer indexes
    V = np.arange(len(V))

    # Threshold probabilites to decide which pairs become edges
    E = np.random.uniform(size=len(edge_probs)) <= edge_probs
    E = edge_indices[E]

    return V, E


def complete_graph(n):
    V = np.arange(n)
    E = np.transpose(np.vstack(np.tril_indices(n)))
    E = E[E[:, 0] != E[:, 1]]

    return V, E


def path_graph(n):
    V = np.arange(n)
    E = []
    for i in range(n-1):
        E.append([i, i+1])
    E = np.array(E)

    return V, E


def cycle_graph(n):
    V = np.arange(n)
    E = []
    for i in range(n-1):
        E.append([i, i+1])
    E.append([n-1, 0])
    E = np.array(E)

    return V, E


def star_graph(n):
    V = np.arange(n+1)
    E = []
    for i in range(1, n+1):
        E.append([0, i])
    E = np.array(E)

    return V, E


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


def create_all_graphs(n):
    """Creates ALL possible isomorphism-unique graphs of size 2 to n"""
    g = []
    # Iterate over vertex counts v
    for v in range(2, n+1):
        K = complete_graph(v)
        # Iterate over all possible graphs of size v
        g_v = []
        edge_selectors = np.array([i for i in product(range(2), repeat=int(v*(v-1)/2))], dtype=bool)
        for es in edge_selectors:
            # Discard if independent set
            if es.sum() == 0:
                continue
            G = (K[0].copy(), K[1][es, :].copy())
            nx_G = np_to_nx(G)
            # Check if new graph is isomorphic with any of the existing ones
            isomorphic = False
            for i in range(len(g_v)):
                if nx.algorithms.isomorphism.is_isomorphic(g_v[i], nx_G):
                    isomorphic = True
                    break

            if not isomorphic:
                g_v.append(nx_G)
        g = g + g_v
    # Convert networkx graph representation to numpy arrays
    g = [nx_to_np(G) for G in g]

    return g
