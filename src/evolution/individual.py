from __future__ import annotations
import numpy as np
from typing import Dict



class Individual:
    learning_rate:float
    entropy_coefficient:float
    value_loss_coefficient:float
    clip_range:float
    imbalance_penalty_lambda:float
    action_smoothness_mu:float
    credit_delta_max:float
    supply_push_max:float
    friction_max:float
    fitness:float
    
    
    def __init__(self, params: Dict[str, float]) -> None:
        #PPO: learning_rate, entropy_coefficient, value_loss_coefficient, clip_range
        #Reward: imbalance_penalty_lambda, action_smoothness_mu
        #Action bounds: credit_delta_max, supply_push_max, friction_max
        
        self.learning_rate = params['learning_rate']
        self.entropy_coefficient = params['entropy_coefficient']
        self.value_loss_coefficient = params['value_loss_coefficient']
        self.clip_range = params['clip_range']        

        self.imbalance_penalty_lambda = params['imbalance_penalty_lambda']
        self.action_smoothness_mu = params['action_smoothness_mu']
        
        self.credit_delta_max = params['credit_delta_max']
        self.supply_push_max = params['supply_push_max']
        self.friction_max = params['friction_max']
        
        self.fitness = float('inf')  # Lower is better (mean volatility)

        
    @classmethod
    def random_init(cls, bounds: Dict[str, Dict[str, Dict[str, float]]], rng: np.random.Generator) -> Individual:
        all_bounds = {}
        all_bounds.update(bounds['ppo_param_bounds'])
        all_bounds.update(bounds['reward_param_bounds'])
        all_bounds.update(bounds['action_bound_bounds'])
        
        #param_mapping = {
        #    'credit_delta_max': 'credit_delta_max',
        #    'supply_push_max': 'supply_push_max',
        #    'friction_max': 'friction_max'
        #}
        
        params = {}
        
        params['learning_rate'] = rng.uniform(
            all_bounds['learning_rate']['min'],
            all_bounds['learning_rate']['max']
        )
        params['entropy_coefficient'] = rng.uniform(
            all_bounds['entropy_coefficient']['min'],
            all_bounds['entropy_coefficient']['max']
        )
        params['value_loss_coefficient'] = rng.uniform(
            all_bounds['value_loss_coefficient']['min'],
            all_bounds['value_loss_coefficient']['max']
        )
        params['clip_range'] = rng.uniform(
            all_bounds['clip_range']['min'],
            all_bounds['clip_range']['max']
        )
        
        params['imbalance_penalty_lambda'] = rng.uniform(
            all_bounds['imbalance_penalty_lambda']['min'],
            all_bounds['imbalance_penalty_lambda']['max']
        )
        params['action_smoothness_mu'] = rng.uniform(
            all_bounds['action_smoothness_mu']['min'],
            all_bounds['action_smoothness_mu']['max']
        )
        
        params['credit_delta_max'] = rng.uniform(
            all_bounds['credit_delta_max']['min'],
            all_bounds['credit_delta_max']['max']
        )
        params['supply_push_max'] = rng.uniform(
            all_bounds['supply_push_max']['min'],
            all_bounds['supply_push_max']['max']
        )
        params['friction_max'] = rng.uniform(
            all_bounds['friction_max']['min'],
            all_bounds['friction_max']['max']
        )
        
        return cls(params)

    
    def get_ppo_params(self) -> Dict[str, float]:
        return {
            'learning_rate': self.learning_rate,
            'ent_coef': self.entropy_coefficient,
            'vf_coef': self.value_loss_coefficient,
            'clip_range': self.clip_range
        }

    
    def get_reward_params(self) -> Dict[str, float]:
        return {
            'imbalance_penalty_lambda': self.imbalance_penalty_lambda,
            'action_smoothness_mu': self.action_smoothness_mu
        }

    
    def get_action_bounds(self) -> Dict[str, float]:
        return {
            'credit_delta_max': self.credit_delta_max,
            'supply_push_max': self.supply_push_max,
            'friction_max': self.friction_max
        }

    
    def get_all_params(self) -> Dict[str, float]:
        return {
            'learning_rate': self.learning_rate,
            'entropy_coefficient': self.entropy_coefficient,
            'value_loss_coefficient': self.value_loss_coefficient,
            'clip_range': self.clip_range,
            'imbalance_penalty_lambda': self.imbalance_penalty_lambda,
            'action_smoothness_mu': self.action_smoothness_mu,
            'credit_delta_max': self.credit_delta_max,
            'supply_push_max': self.supply_push_max,
            'friction_max': self.friction_max
        }

    
    def set_fitness(self, fitness: float) -> None:
        self.fitness = fitness

        
    def copy(self) -> Individual:
        params = self.get_all_params()
        new_ind = Individual(params)
        new_ind.fitness = self.fitness
        return new_ind
