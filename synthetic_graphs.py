from itertools import product
import random
import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


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
        ((x <= 0.5) & (y <= 0.5))
        | ((x > 0.5) & (y > 0.5))
    ).astype(float)


def graphon_big_qlique(x, y):
    """Complete bipartite graphon"""
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


def visualize_graphon(W, resolution=300, ax=plt):
    """Plots the graphon W as a greyscale image. The darker the closer to 1."""
    uniform_args = np.linspace(start=0, stop=1, num=resolution)
    cartesian_product = np.transpose(
        [np.tile(uniform_args, len(uniform_args)),
         np.repeat(uniform_args, len(uniform_args))]
    )
    img_mat = (
        W(cartesian_product[:, 0], cartesian_product[:, 1])
        .reshape(resolution, resolution)
    )
    plt.gray()
    ax.imshow(
        X=img_mat,
        origin='lower',
        extent=[0, 1, 0, 1],
        cmap='plasma',
        vmin=0,
        vmax=1
    )
    ax.colorbar()


def t_discrete(g, G, epsilon=0.01, gamma=0.95):
    """Homomorphism density for finite graphs estimation with naive Monte-Carlo"""
    # Unpack graph structs
    V_g, E_g = g
    V_G, E_G = G

    # Create adjacency matrix
    A_G = np.zeros((len(V_G), len(V_G)), dtype=int)
    A_G[E_G[:,0], E_G[:,1]] = 1
    A_G[E_G[:,1], E_G[:,0]] = 1

    # Sample size
    N = math.ceil((math.log(2) - math.log(1 - gamma)) / (2*epsilon**2))

    # Sample vertex mappings V_g -> V_G
    mappings = np.random.randint(low=0, high=len(V_G)-1, size=(N, len(V_g)))

    # Create mapped vertex edges to later check in adjacency matrix
    mapped_edges = np.empty(shape=(len(E_g)*N, 2), dtype=np.uint16)
    for i in range(N):
        mapped_edges[i*len(E_g):(i+1)*len(E_g)] = mappings[i][E_g]

    # Get adjacency for each edge of the random mappings
    adjacency_indicators = A_G[mapped_edges[:,0], mapped_edges[:,1]]

    # Multiply adjacencies to check if the mappings represent homomorphisms then average
    hom_density = (
        adjacency_indicators
        .reshape(len(E_g), N)
        .prod(axis=0)
        .sum()
        / N
    )

    return hom_density


def complete_graph(n):
    V = np.arange(n)
    E = np.transpose(np.vstack(np.tril_indices(n)))
    E = E[E[:,0] != E[:,1]]

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
            G = (K[0].copy(), K[1][es ,:].copy())
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
