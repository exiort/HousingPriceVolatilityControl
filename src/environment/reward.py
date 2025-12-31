from __future__ import annotations
import numpy as np
from typing import Optional



class RewardFunction:
    lambda_imbalance:float
    mu_smoothness:float
    prev_action:Optional[np.ndarray]
    
    def __init__(self, lambda_imbalance: float, mu_smoothness: float) -> None:
        #R_t = −σ_t − λ · |z_t| − μ · ||a_t − a_{t-1}||₂
        
        self.lambda_imbalance = lambda_imbalance
        self.mu_smoothness = mu_smoothness
        self.prev_action = None

        
    def reset(self) -> None:
        self.prev_action = None

        
    def compute_reward(
        self,
        rolling_volatility: float,
        imbalance: float,
        action: np.ndarray
    ) -> float:
        #rolling_volatility: Current rolling volatility σ_t
        #imbalance: Supply-demand imbalance z_t
        #action: Current action vector [a_c, a_s, a_f]
        #to 
        #Reward value
        
        volatility_penalty = rolling_volatility
        
        imbalance_penalty = self.lambda_imbalance * np.abs(imbalance)
        
        if self.prev_action is None:
            smoothness_penalty = 0.0
        else:
            action_diff = action - self.prev_action
            smoothness_penalty = self.mu_smoothness * np.linalg.norm(action_diff, ord=2)
        
        self.prev_action = action.copy()
        
        reward = -volatility_penalty - imbalance_penalty - smoothness_penalty
        
        return reward
