from __future__ import annotations
import numpy as np



class MarketDynamics:
    alpha_0:float
    alpha_1:float
    alpha_2:float
    trend_eta:float
    beta:float
    gemma:float
    noise_std:float
    supply_lag:int

    
    def __init__(
        self,
        alpha_0: float,
        alpha_1: float,
        alpha_2: float,
        trend_eta: float,
        beta: float,
        gamma: float,
        noise_std: float,
        supply_lag: int
    ) -> None:        
        #alpha_0: Demand baseline
        #alpha_1: Demand sensitivity to expected return
        #alpha_2: Demand sensitivity to credit tightness
        #trend_eta: Trend extrapolation coefficient
        #beta: Supply response coefficient
        #gamma: Price adjustment speed
        #noise_std: Price shock standard deviation
        #supply_lag: Supply pipeline lag (k)
      
        self.alpha_0 = alpha_0
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.trend_eta = trend_eta
        self.beta = beta
        self.gamma = gamma
        self.noise_std = noise_std
        self.supply_lag = supply_lag

        
    def compute_expected_return(self, prev_return: float) -> float:
        #E[r_t] = η · r_{t-1}

        return self.trend_eta * prev_return

    
    def compute_demand(self, expected_return: float, credit_tightness: float) -> float:
        #D_t = α₀ + α₁ · E[r_t] − α₂ · c_t
        
        return self.alpha_0 + self.alpha_1 * expected_return - self.alpha_2 * credit_tightness

    
    def compute_supply(self, pipeline_lagged: float) -> float:    
        #S_t = β · π_{t-k}

        return self.beta * pipeline_lagged

    
    def compute_imbalance(self, demand: float, supply: float) -> float:    
        #z_t = D_t − S_t

        return demand - supply

    
    def apply_market_friction(self, imbalance: float, friction_action: float) -> float:
        #ẑ_t = z_t · (1 − a_f)
        
        return imbalance * (1.0 - friction_action)

    
    def compute_price_update(
        self,
        current_log_price: float,
        adjusted_imbalance: float,
        rng: np.random.Generator
    ) -> float:
        #p_{t+1} = p_t + γ · ẑ_t + ε_t
        #ε_t ~ N(0, σ_ε²)
        
        noise = rng.normal(0.0, self.noise_std)
        return current_log_price + self.gamma * adjusted_imbalance + noise

    
    def compute_credit_update(self, current_credit: float, credit_action: float) -> float:
        #c_{t+1} = c_t + a_c
        
        return current_credit + credit_action

    
    def update_supply_pipeline(self, current_pipeline: float, supply_action: float) -> float:    
        #π_t = π_{t-1} + a_s

        return current_pipeline + supply_action
