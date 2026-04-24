import numpy as np
from model import ConnectFourNet
from game_runner import evaluate

def create_population(size):
    """Creates a population of randomly initialised models."""
    return [ConnectFourNet() for _ in range(size)]


def score_population(population, games_per_opponent=6):
    """
    Scores every model by evaluating it against a random sample of others.
    Returns a list of (score, model) tuples sorted best first.
    """
    sample_size = min(10, len(population) - 1)
    scored = []

    for i, model in enumerate(population):
        opponents = [population[j] for j in range(len(population)) if j != i]
        opponents = np.random.choice(opponents, size=sample_size, replace=False).tolist()
        score = evaluate(model, opponents, games_per_opponent)
        scored.append((score, model))

    return sorted(scored, key=lambda x: x[0], reverse=True)


def select_elites(scored_population, elite_frac=0.2):
    """
    Keeps the top elite_frac of the population unchanged.
    Returns a list of elite models.
    """
    n_elites = max(2, int(len(scored_population) * elite_frac))
    return [model for _, model in scored_population[:n_elites]]


def crossover(parent_a, parent_b):
    """
    Creates a child model by randomly mixing weights from two parents
    at the layer level — each layer's weights come entirely from one parent.
    This preserves layer structure better than per-weight crossover.
    """
    child = ConnectFourNet()
    params_a = list(parent_a.parameters())
    params_b = list(parent_b.parameters())
    child_params = list(child.parameters())

    for i in range(len(child_params)):
        source = params_a[i] if np.random.rand() > 0.5 else params_b[i]
        child_params[i].data.copy_(source.data)

    return child


def mutate(model, mutation_rate=0.05, mutation_strength=0.1):
    """
    Randomly perturbs a fraction of weights by adding gaussian noise.
    
    mutation_rate     — probability each weight is mutated
    mutation_strength — std dev of the gaussian noise added
    """
    weights = model.get_weights()
    mask = np.random.rand(len(weights)) < mutation_rate
    noise = np.random.randn(len(weights)) * mutation_strength
    weights[mask] += noise[mask]
    model.set_weights(weights)
    return model


def breed_new_generation(elites, target_size, mutation_rate=0.05, mutation_strength=0.1):
    """
    Fills the population back up to target_size by:
      - Keeping all elites unchanged
      - Breeding crossed-over children from elite pairs
      - Adding a small number of random newcomers for diversity
    """
    new_population = elites.copy()
    n_newcomers = max(1, int(target_size * 0.05))  # 5% fresh random models
    n_children = target_size - len(elites) - n_newcomers

    # Breed children
    for _ in range(n_children):
        parent_a, parent_b = np.random.choice(elites, size=2, replace=False)
        child = crossover(parent_a, parent_b)
        child = mutate(child, mutation_rate, mutation_strength)
        new_population.append(child)

    # Add newcomers
    for _ in range(n_newcomers):
        new_population.append(ConnectFourNet())

    return new_population


def next_generation(population, elite_frac=0.2, mutation_rate=0.05,
                    mutation_strength=0.1, games_per_opponent=6):
    """
    Convenience function that runs a full generation cycle:
    score → select → breed → return new population + best score
    """
    scored = score_population(population, games_per_opponent)
    best_score = scored[0][0]
    elites = select_elites(scored, elite_frac)
    new_pop = breed_new_generation(elites, len(population), mutation_rate, mutation_strength)
    return new_pop, best_score, elites[0]  # population, best score, best model
