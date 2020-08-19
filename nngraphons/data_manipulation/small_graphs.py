from itertools import product

import numpy as np
import networkx as nx

from nngraphons.data_manipulation.networkx_conversion import np_to_nx, nx_to_np


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
