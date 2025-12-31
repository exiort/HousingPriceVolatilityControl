from __future__ import annotations
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import Dict
import gymnasium as gym
import torch.nn as nn



class PPOTrainer:
    env:gym.Env
    ppo_params:Dict[str, float]
    rl_seed:int
    total_timesteps:int

    
    def __init__(
        self,
        env: gym.Env,
        ppo_params: Dict[str, float],
        rl_seed: int,
        total_timesteps: int
    ):
        #env: Gymnasium environment
        #ppo_params: PPO hyperparameters (learning_rate, ent_coef, vf_coef, clip_range)
        #rl_seed: Random seed for RL
        #total_timesteps: Total training timesteps
    
        self.env = env
        self.ppo_params = ppo_params
        self.rl_seed = rl_seed
        self.total_timesteps = total_timesteps
        
        vec_env = DummyVecEnv([lambda: env])
        
        # Create PPO model with fixed architecture: MLP 2x64, tanh
        self.model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=ppo_params['learning_rate'],
            ent_coef=ppo_params['ent_coef'],
            vf_coef=ppo_params['vf_coef'],
            clip_range=ppo_params['clip_range'],
            policy_kwargs={
                "net_arch": [64, 64], 
                "activation_fn": nn.Tanh
            },
            seed=rl_seed,
            device='cpu', #faster than GPU for small networks
            verbose=0,
            n_steps=240,
            batch_size=240,
            n_epochs=5
        )

        
    def train(self) -> PPO:
        self.model.learn(total_timesteps=self.total_timesteps)
        return self.model

    
    def get_model(self) -> PPO:
        return self.model
