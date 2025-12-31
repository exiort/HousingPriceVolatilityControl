from __future__ import annotations
import numpy as np
from collections import deque
from typing import Deque



class StateManager:
    volatility_window:int
    return_history:Deque[float]

    
    def __init__(self, volatility_window: int) -> None:
        #volatility_window: Window size W for rolling volatility
        
        self.volatility_window = volatility_window
        self.return_history = deque(maxlen=volatility_window)

        
    def reset(self) -> None:
        self.return_history.clear()

        
    def add_return(self, log_return: float) -> None:
        self.return_history.append(log_return)

        
    def compute_rolling_volatility(self) -> float:
        #σ_t = sqrt( (1/W) · Σ ( r_{t-i} − r̄_t )² )
        
        if len(self.return_history) == 0:
            return 0.0
        
        returns_array = np.array(self.return_history)
        mean_return = np.mean(returns_array)
        variance = np.mean((returns_array - mean_return) ** 2)
        return np.sqrt(variance)

    
    def get_num_returns(self) -> int:
        return len(self.return_history)
