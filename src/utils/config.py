from __future__ import annotations
import json
from typing import Dict, Any



def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    _validate_config(config)
    return config


def _validate_config(config: Dict[str, Any]) -> None:
    required_sections = [
        'time', 'price_dynamics', 'demand', 'supply', 'training',
        'ppo_param_bounds', 'reward_param_bounds', 'action_bound_bounds',
        'evolution', 'seeds'
    ]
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    
    time_params = ['episode_length', 'volatility_window', 'supply_lag']
    for param in time_params:
        if param not in config['time']:
            raise ValueError(f"Missing time parameter: {param}")
    
    seed_params = ['env_train_seed', 'env_eval_seeds', 'rl_seed', 'ec_seed']
    for param in seed_params:
        if param not in config['seeds']:
            raise ValueError(f"Missing seed parameter: {param}")
    
    if not isinstance(config['seeds']['env_eval_seeds'], list):
        raise ValueError("env_eval_seeds must be a list")


def get_time_config(config: Dict[str, Any]) -> Dict[str, int]:
    return config['time']


def get_price_dynamics_config(config: Dict[str, Any]) -> Dict[str, float]:
    return config['price_dynamics']


def get_demand_config(config: Dict[str, Any]) -> Dict[str, float]:
    return config['demand']


def get_supply_config(config: Dict[str, Any]) -> Dict[str, float]:
    return config['supply']


def get_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return config['training']


def get_ppo_param_bounds(config: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    return config['ppo_param_bounds']


def get_reward_param_bounds(config: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    return config['reward_param_bounds']


def get_action_bound_bounds(config: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    return config['action_bound_bounds']


def get_evolution_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return config['evolution']


def get_seeds(config: Dict[str, Any]) -> Dict[str, Any]:
    return config['seeds']
