import numpy as np


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
