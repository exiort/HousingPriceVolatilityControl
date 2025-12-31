from __future__ import annotations
import json
from typing import Dict, Any, List



class SeedManager:
    env_train_see:int
    env_eval_seeds:List[int]
    rl_seed:int
    ec_seed:int
    
    
    def __init__(self, seeds_config: Dict[str, Any]) -> None:
        self.env_train_seed = seeds_config['env_train_seed']
        self.env_eval_seeds = seeds_config['env_eval_seeds']
        self.rl_seed = seeds_config['rl_seed']
        self.ec_seed = seeds_config['ec_seed']
        
        if not isinstance(self.env_eval_seeds, List):
            raise ValueError("env_eval_seeds must be a list")
        if len(self.env_eval_seeds) == 0:
            raise ValueError("env_eval_seeds must not be empty")

        
    def get_env_train_seed(self) -> int:
        return self.env_train_seed

    
    def get_env_eval_seeds(self) -> List[int]:
        return self.env_eval_seeds

    
    def get_rl_seed(self) -> int:
        return self.rl_seed

    
    def get_ec_seed(self) -> int:
        return self.ec_seed

        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'env_train_seed': self.env_train_seed,
            'env_eval_seeds': self.env_eval_seeds,
            'rl_seed': self.rl_seed,
            'ec_seed': self.ec_seed
        }

    
    def save_to_file(self, filepath: str) -> None:
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
