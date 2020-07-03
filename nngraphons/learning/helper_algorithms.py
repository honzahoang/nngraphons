import numpy as np
from deap import base, creator, tools
from deap.algorithms import eaSimple


def min_norm_in_polytope_GA(
    points,
    population_size=50,
    n_generations=500,
    cross_over_probability=0.5,
    mutation_probability=0.2,
    tournament_size=5
):
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
