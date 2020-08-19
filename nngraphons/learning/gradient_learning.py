from typing import List, Tuple, Callable, Optional
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from deap import base, creator, tools
from deap.algorithms import eaSimple
from IPython.display import clear_output

from nngraphons.visualization.graphon_visualization import visualize_pytorch_net_graphon
from nngraphons.learning.homomorphism_density import t_nn


def gradient_descent(
    g: List[Tuple],
    net: nn.Module,
    n: int,
    t_g: List[float],
    L: Callable[[List[float], List[torch.Tensor]], float],
    stopping_criterion: Callable[[float], bool],
    optimizer: Optional[torch.optim.Optimizer] = None
) -> None:
    """
    Iteratively updates net's parameters in the gradient direction w.r.t L

    Parameters
    ----------
    g : List[Tuple]
        List of small graphs for which to match homomorphism densities of the
        network and training graph
    net : torch.nn.Module
        PyTorch neural network to represent graphon
    n : int
        Number of samples to use for density approximation at each training iteration
    t_g : List[float]
        List of ground truth homomorphism densities of small graphs g
    L : Callable[[List[float], List[torch.Tensor]], float]
        Loss function accepting list of ground truth homomorphism densities and netowrk densities
    stopping_criterion : Callable[[float], bool]
        Training stopping criterion function accepting loss function value
    optimizer : torch.optim.Optimizer, optional
        Optimizer to use, Adam is used by default
    """
    # Create optimizer
    if optimizer is None:
        optimizer = optim.Adam(net.parameters())

    # Training loop
    while True:
        # Zero out old accumulated gradients
        optimizer.zero_grad()

        # Calculate homomorphism densities w.r.t network
        t_net = [t_nn(F, net, n, True) for F in g]

        # Calculate loss value
        loss = L(t_g, t_net)

        # Progress print
        clear_output(wait=True)
        print(f'Current loss: {loss.item()}')

        # Visualize progress
        visualize_pytorch_net_graphon(net)

        # Stopping criterion
        if stopping_criterion(loss.item()):
            break

        # Backprop gradients
        loss.backward()

        # Update weights
        optimizer.step()


def min_norm_in_polytope_GA(
    points: np.ndarray,
    population_size=50,
    n_generations=500,
    cross_over_probability=0.5,
    mutation_probability=0.2,
    tournament_size=5
) -> np.ndarray:
    """Finds minimum norm element in convex hull of points using a genetic algorithm (GA).

    Parameters
    ----------
    points : np.ndarray
        Matrix of size n x D, where n is the number of points and D is the dimensionality of the
        points
    population_size : int, optional
        Population size for the GA, by default 50
    n_generations : int, optional
        Number of generations for the GA, by default 500
    cross_over_probability : float, optional
        Cross over probability of two individuals, by default 0.5
    mutation_probability : float, optional
        Mutation probability of an individual, by default 0.2
    tournament_size : int, optional
        Number of individuals carrying on to the next generation, by default 5

    Returns
    -------
    np.ndarray
        Point in the convex hull of points with an approximately minimal L2 norm
    """
    def random_convex_coeffs(icls):
        alpha = np.random.uniform(size=points.shape[0])
        alpha = alpha / alpha.sum()
        return icls(alpha)

    def convex_cross_over(alpha, beta, icls):
        coeff1, coeff2 = np.random.uniform(size=2)
        ind1 = coeff1 * alpha + (1 - coeff1) * beta
        ind2 = coeff2 * alpha + (1 - coeff2) * beta
        return icls(ind1), icls(ind2)

    def convex_mutation_normal(alpha, icls, loc=0, scale=1, prob=0.01):
        mutation = np.random.normal(loc=loc, scale=scale, size=alpha.shape)
        mutation[
            (np.random.uniform(size=alpha.shape) > prob)
            | (mutation < 0)
        ] = 0.0
        mutant = alpha + mutation
        mutant = mutant / mutant.sum()
        return (icls(mutant),)

    def L2(alpha):
        convex_combination = np.matmul(alpha, points)
        return (np.linalg.norm(convex_combination),)

    creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
    creator.create('Individual', np.ndarray, fitness=creator.FitnessMin)

    tb = base.Toolbox()
    tb.register('individual', random_convex_coeffs, icls=creator.Individual)
    tb.register('population', tools.initRepeat, list, tb.individual)
    tb.register('mate', convex_cross_over, icls=creator.Individual)
    tb.register('mutate', convex_mutation_normal, icls=creator.Individual)
    tb.register('select', tools.selTournament, tournsize=tournament_size)
    tb.register('evaluate', L2)

    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register('avg', np.mean)
    stats.register('std', np.std)
    stats.register('min', np.min)
    stats.register('max', np.max)
    pop = tb.population(n=population_size)
    final_pop, logbook = eaSimple(
        population=pop,
        toolbox=tb,
        cxpb=cross_over_probability,
        mutpb=mutation_probability,
        ngen=n_generations,
        stats=stats,
        verbose=False
    )

    min_norm_coeffs = tools.selBest(final_pop, 1)[0]
    return np.array(np.matmul(min_norm_coeffs, points))


def multiple_gradient_descent_algorithm(
    g,
    net,
    n,
    t_g,
    L
) -> None:
    # Initialize optimizer
    optimizer = torch.optim.Adam(params=net.parameters())
    # Training loop
    losses = []
    omega_norms = []
    while True:
        # For each finite graph g[i] calculate the squared homomorphism density loss w.r.t it and
        # save the parameter gradients J[i]
        J = []
        t_nets = []
        print('Calculating homomorphism densities')
        for i in range(len(g)):
            # Zero out old gradients
            net.zero_grad()
            # Compute finite graph specific loss
            t_net = t_nn(g[i], net, n, True)
            t_nets.append(t_net)
            loss = (t_g[i] - t_net)**2
            # Compute new gradients w.r.t computed loss value
            loss.backward()
            # Save loss specific parameter gradient as 1D vector
            param_gradients = [
                p.grad.clone().detach().cpu().numpy().flatten()
                for p in net.parameters()
            ]
            J.append(np.concatenate(param_gradients))
        J = np.vstack(J)

        # Determine direction of descent to Pareto-stationary point
        print('Finding minimum norm')
        omega = min_norm_in_polytope_GA(J, 25, 1000)
        omega_norms.append(np.linalg.norm(omega))

        clear_output(wait=True)
        # If omega norm is small enough, we are already in a pareto-stationary configuration
        if np.array(omega_norms[:-3]).mean() < 0.0001:
            # Set omega to gradients of finite graph g[i] with the maximum squared loss
            omega = J[np.argmax(np.array(losses)), :]

        print('Applying gradient')
        # Set the network gradient to omega
        pos = 0
        for p in net.parameters():
            param_shape = np.array(p.grad.size())
            param_length = param_shape.prod()
            p.grad.data = (
                torch
                .from_numpy(omega[pos:pos+param_length].reshape(param_shape))
                .float()
                .to(os.environ['COMPUTATION_DEVICE'])
            )
            pos += param_length

        # Update weights
        optimizer.step()

        # Print loss
        with torch.no_grad():
            print(f'Omega norm size: {np.linalg.norm(omega)}')
            loss = L(t_g, t_nets)
            print(f'Current loss: {loss.item()}')
            losses.append(loss.item())

        # Visualize progress
        visualize_pytorch_net_graphon(net)
