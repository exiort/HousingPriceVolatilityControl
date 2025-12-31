import numpy as np
from typing import List, Dict, Any

from .individual import Individual
from .genetic_ops import elitism_selection, tournament_selection, gaussian_mutation



class Population:
    population_size:int
    bounds:Dict[str, Dict[str, Dict[str, float]]]

    
    def __init__(
        self,
        population_size: int,
        bounds: Dict[str, Dict[str, Dict[str, float]]],
        evolution_config: Dict[str, Any],
        rng: np.random.Generator
    ) -> None:
        #population_size: Number of individuals
        #bounds: Parameter bounds (ppo_param_bounds, reward_param_bounds, action_bound_bounds)
        #evolution_config: Evolution settings (elite_fraction, tournament_size, mutation_std)
        #rng: Random number generator
       
        self.population_size = population_size
        self.bounds = bounds
        self.elite_fraction = evolution_config['elite_fraction']
        self.tournament_size = evolution_config['tournament_size']
        self.mutation_stds = evolution_config['mutation_std']
        self.rng = rng
        
        self.individuals: List[Individual] = []
        for _ in range(population_size):
            ind = Individual.random_init(bounds, rng)
            self.individuals.append(ind)

            
    def get_individuals(self) -> List[Individual]:
        return self.individuals

    
    def get_best_individual(self) -> Individual:
        return min(self.individuals, key=lambda ind: ind.fitness) #lowest_best

    
    def evolve_generation(self) -> None:
        """
        Perform one generation step using (μ+λ) strategy.
        
        Steps:
        1. Select elites
        2. Generate offspring using tournament selection + mutation
        3. Combine elites and offspring
        4. Keep best population_size individuals
        """
        # 1. Select elites
        elites = elitism_selection(self.individuals, self.elite_fraction)
        
        # 2. Generate offspring
        num_offspring = self.population_size
        offspring = []
        
        for _ in range(num_offspring):
            parent = tournament_selection(self.individuals, self.tournament_size, self.rng)
            child = gaussian_mutation(parent, self.mutation_stds, self.bounds, self.rng)
            offspring.append(child)
        
        # 3. Combine elites and offspring (μ+λ)
        combined = elites + offspring
        
        # 4. Select best population_size individuals for next generation
        combined_sorted = sorted(combined, key=lambda ind: ind.fitness)
        self.individuals = combined_sorted[:self.population_size]
