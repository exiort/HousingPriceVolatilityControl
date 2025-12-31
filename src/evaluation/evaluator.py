from __future__ import annotations
import numpy as np
from typing import Dict, Any, List, Tuple

from src.environment.housing_env import HousingVolatilityEnv
from src.training.ppo_trainer import PPOTrainer
from src.evolution.individual import Individual
from src.utils.seeding import SeedManager 
from src.utils.config import (
    get_time_config, get_price_dynamics_config,
    get_demand_config, get_supply_config, get_training_config
)
        


class Evaluator:
    config:Dict[str, Any]
    seed_manager:SeedManager

    
    def __init__(self, config: Dict[str, Any], seed_manager:SeedManager) -> None:
        #config: Full configuration dictionary
        #seed_manager: Seed manager
        
        self.config = config
        self.seed_manager = seed_manager

        
    def evaluate_best_individual(
        self,
        best_individual: Individual,
        num_episodes_per_seed: int = 10
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        #best_individual: Best individual from training
        #num_episodes_per_seed: Episodes to run per seed
        #to
        #Tuple of (evaluation_summary, evaluation_timeseries)
        #- evaluation_summary: List of {eval_seed, mean_rolling_vol}
        #- evaluation_timeseries: List of trajectory data with eval_seed
                
        time_config = get_time_config(self.config)
        price_dynamics_config = get_price_dynamics_config(self.config)
        demand_config = get_demand_config(self.config)
        supply_config = get_supply_config(self.config)
        training_config = get_training_config(self.config)
        
        ppo_params = best_individual.get_ppo_params()
        reward_params = best_individual.get_reward_params()
        action_bounds = best_individual.get_action_bounds()
        
        eval_seeds = self.seed_manager.get_env_eval_seeds()
        rl_seed = self.seed_manager.get_rl_seed()
        
        evaluation_summary = []
        evaluation_timeseries = []
        
        for eval_seed in eval_seeds:
            print(f"Evaluating on seed {eval_seed}")
            
            # Create environment with this eval seed
            env = HousingVolatilityEnv(
                time_config=time_config,
                price_dynamics_config=price_dynamics_config,
                demand_config=demand_config,
                supply_config=supply_config,
                action_bounds=action_bounds,
                reward_params=reward_params,
                env_seed=eval_seed
            )
            
            trainer = PPOTrainer(
                env=env,
                ppo_params=ppo_params,
                rl_seed=rl_seed,
                total_timesteps=training_config['timesteps']
            )
            model = trainer.train()
            
            # Evaluate
            seed_volatilities = []
            
            for episode_idx in range(num_episodes_per_seed):
                obs, info = env.reset()
                episode_volatilities = []
                timestep = 0
                
                done = False
                while not done:
                    # Get action
                    action, _ = model.predict(obs, deterministic=True)
                    
                    # Step
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    
                    if episode_idx == 0:
                        timeseries_entry = {
                            'eval_seed': eval_seed,
                            'episode': episode_idx,
                            'timestep': timestep,
                            'log_price': info['log_price'],
                            'return': info['return'],
                            'rolling_volatility': info['rolling_volatility'],
                            'imbalance': info['imbalance'],
                            'credit_tightness': info['credit_tightness'],
                            'action_credit_delta': float(action[0]),
                            'action_supply_push': float(action[1]),
                            'action_friction': float(action[2]),
                            'reward': float(reward)
                        }
                        evaluation_timeseries.append(timeseries_entry)
                    
                    episode_volatilities.append(info['rolling_volatility'])
                    obs = next_obs
                    timestep += 1
                
                
                seed_volatilities.append(np.mean(episode_volatilities))
            
            # Compute mean volatility for this seed
            mean_vol = np.mean(seed_volatilities)
            
            # Add to summary
            summary_entry = {
                'eval_seed': eval_seed,
                'mean_rolling_vol': mean_vol
            }
            evaluation_summary.append(summary_entry)
            
            print(f"  Mean rolling volatility: {mean_vol:.6f}")
        
        return evaluation_summary, evaluation_timeseries
