from __future__ import annotations
import numpy as np
from typing import Dict, Any, Tuple, List

from .ppo_trainer import PPOTrainer

from src.environment.housing_env import HousingVolatilityEnv
from src.evolution.individual import Individual
from src.utils.config import (
    get_time_config, get_price_dynamics_config,
    get_demand_config, get_supply_config, get_training_config
)



def evaluate_individual(
    individual: Individual,
    config: Dict[str, Any],
    env_seed: int,
    rl_seed: int,
    num_eval_episodes: int = 1
) -> Tuple[float, List[Dict[str, Any]]]:
    #individual: Individual to evaluate
    #config: Full configuration dictionary
    #env_seed: Environment seed (env_train_seed for training)
    #rl_seed: RL algorithm seed
    #num_eval_episodes: Number of episodes to evaluate

    time_config = get_time_config(config)
    price_dynamics_config = get_price_dynamics_config(config)
    demand_config = get_demand_config(config)
    supply_config = get_supply_config(config)
    training_config = get_training_config(config)
    
    ppo_params = individual.get_ppo_params()
    reward_params = individual.get_reward_params()
    action_bounds = individual.get_action_bounds()
    
    env = HousingVolatilityEnv(
        time_config=time_config,
        price_dynamics_config=price_dynamics_config,
        demand_config=demand_config,
        supply_config=supply_config,
        action_bounds=action_bounds,
        reward_params=reward_params,
        env_seed=env_seed
    )
    
    # Train PPO
    trainer = PPOTrainer(
        env=env,
        ppo_params=ppo_params,
        rl_seed=rl_seed,
        total_timesteps=training_config['timesteps']
    )
    model = trainer.train()
    
    total_volatility = 0.0
    trajectory_data = []
    
    for episode_idx in range(num_eval_episodes):
        obs, info = env.reset()
        episode_trajectory = []
        episode_volatilities = []
        
        done = False
        while not done:
            # Get action from trained policy
            action, _ = model.predict(obs, deterministic=True)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            step_data = {
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
            episode_trajectory.append(step_data)
            episode_volatilities.append(info['rolling_volatility'])
            
            obs = next_obs
        
        # Compute mean volatility for this episode
        episode_mean_vol = float(np.mean(episode_volatilities))
        total_volatility += episode_mean_vol
        
        if episode_idx == 0:
            trajectory_data = episode_trajectory
    
    fitness = total_volatility / num_eval_episodes
    
    return fitness, trajectory_data
