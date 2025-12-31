from __future__ import annotations
import numpy as np
from typing import Dict, Any, Tuple, List

from .fitness import evaluate_individual

from src.evolution.population import Population
from src.evolution.individual import Individual
from src.utils.seeding import SeedManager
from src.utils.config import (
    get_ppo_param_bounds, get_reward_param_bounds,
    get_action_bound_bounds, get_evolution_config
)



class TrainingPipeline:
    config:Dict[str, Any]
    seed_manager:SeedManager
    
    evolution_config:Dict[str, Any]
    bounds:Dict[str, Dict[str, Dict[str, float]]]

    ec_rng:np.random.Generator
    population:Population

    generation_counter:int
    all_generations_data:List[Dict[str, Any]]

    
    def __init__(
        self,
        config: Dict[str, Any],
        seed_manager: SeedManager
    ):
        #config: Full configuration dictionary
        #seed_manager: Seed manager
        
        self.config = config
        self.seed_manager = seed_manager
        
        self.evolution_config = get_evolution_config(config)
        
        self.bounds = {
            'ppo_param_bounds': get_ppo_param_bounds(config),
            'reward_param_bounds': get_reward_param_bounds(config),
            'action_bound_bounds': get_action_bound_bounds(config)
        }
        
        self.ec_rng = np.random.default_rng(seed_manager.get_ec_seed())
        
        self.population = Population(
            population_size=self.evolution_config['population_size'],
            bounds=self.bounds,
            evolution_config=self.evolution_config,
            rng=self.ec_rng
        )
        
        self.generation_counter = 0
        self.all_generations_data = []

        
    def run_evolution(self, num_generations: int) -> Tuple[Individual, List[Dict[str, Any]]]: 
        #num_generations: Number of generations to run
        #to
        #Tuple of (best_individual, training_data)
        #   - best_individual: Best individual found
        #   - training_data: Training summary data for all generations
       
        env_train_seed = self.seed_manager.get_env_train_seed()
        rl_seed = self.seed_manager.get_rl_seed()
        
        for gen in range(num_generations):
            print(f"Generation {gen + 1}/{num_generations}")
            
            for idx, individual in enumerate(self.population.get_individuals()):
                print(f"  Evaluating individual {idx + 1}/{self.evolution_config['population_size']}")
                
                # Evaluate fitness
                fitness, _ = evaluate_individual(
                    individual=individual,
                    config=self.config,
                    env_seed=env_train_seed,
                    rl_seed=rl_seed,
                    num_eval_episodes=5
                )
                
                individual.set_fitness(fitness)
                
                individual_data = {
                    'generation': gen,
                    'individual_id': idx,
                    'fitness': fitness,
                    **individual.get_all_params()
                }
                self.all_generations_data.append(individual_data)
            
            best_ind = self.population.get_best_individual()
            print(f"  Best fitness: {best_ind.fitness:.6f}")
            
            if gen < num_generations - 1:
                self.population.evolve_generation()
            
            self.generation_counter += 1
        
        final_best = self.population.get_best_individual()
        return final_best, self.all_generations_data
