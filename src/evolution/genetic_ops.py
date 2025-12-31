from __future__ import annotations
import numpy as np
from typing import List, Dict

from .individual import Individual



def elitism_selection(population: List[Individual], elite_fraction: float) -> List[Individual]:
    # Sort by fitness (lower is better)
    sorted_pop = sorted(population, key=lambda ind: ind.fitness)
    
    # Select top fraction
    num_elite = max(1, int(len(population) * elite_fraction))
    elites = sorted_pop[:num_elite]
    
    return elites


def tournament_selection(
    population: List[Individual],
    tournament_size: int,
    rng: np.random.Generator
) -> Individual:
    #population: List of individuals
    #tournament_size: Number of individuals in tournament
    #rng: Random number generator
        
    # Randomly select tournament_size individuals
    tournament_indices = rng.choice(len(population), size=tournament_size, replace=False)
    tournament = [population[i] for i in tournament_indices]
    
    # Return best (lowest fitness)
    winner = min(tournament, key=lambda ind: ind.fitness)
    return winner


def gaussian_mutation(
    individual: Individual,
    mutation_stds: Dict[str, float],
    bounds: Dict[str, Dict[str, Dict[str, float]]],
    rng: np.random.Generator
) -> Individual:
    #individual: Individual to mutate
    #mutation_stds: Standard deviation for each parameter
    #bounds: Parameter bounds for clipping
    #rng: Random number generator

    
    params = individual.get_all_params()
    all_bounds = {}
    all_bounds.update(bounds['ppo_param_bounds'])
    all_bounds.update(bounds['reward_param_bounds'])
    all_bounds.update(bounds['action_bound_bounds'])
    
    mutated_params = {}
    for param_name, param_value in params.items():
        std = mutation_stds.get(param_name, 0.0)
        noise = rng.normal(0.0, std)
        new_value = param_value + noise
        
        min_val = all_bounds[param_name]['min']
        max_val = all_bounds[param_name]['max']
        new_value = np.clip(new_value, min_val, max_val)
        
        mutated_params[param_name] = new_value
    
    mutated_individual = Individual(mutated_params)
    
    return mutated_individual
