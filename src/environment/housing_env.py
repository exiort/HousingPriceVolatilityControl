from __future__ import annotations
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from collections import deque
from typing import Deque, Dict, Any, Tuple, Optional

from .dynamics import MarketDynamics
from .state import StateManager
from .reward import RewardFunction



class HousingVolatilityEnv(gym.Env):
    #State: [p_t, r_t, σ_t, z_t, c_t]
    #Action: [a_c, a_s, a_f]  
    
    metadata = {"render_modes": []}

    episode_length:int
    volatility_window:int
    supply_lag:int
    env_seed:int

    max_credit_delta:float
    max_supply_push:float
    max_friction:float

    dynamics:MarketDynamics
    state_manager:StateManager
    reward_fn:RewardFunction

    rng:np.random.Generator
    observation_space:gym.Space[np.ndarray]
    action_space:gym.Space[np.ndarray]

    log_price:float
    log_return:float
    credit_tightness:float
    imbalance:float
    supply_pipeline:float
    pipeline_history:Deque[float]

    current_step:int
    
    def __init__(
        self,
        time_config: Dict[str, int],
        price_dynamics_config: Dict[str, float],
        demand_config: Dict[str, float],
        supply_config: Dict[str, float],
        action_bounds: Dict[str, float],
        reward_params: Dict[str, float],
        env_seed: int
    ) -> None:
        #time_config: Episode length, volatility window, supply lag
        #price_dynamics_config: Gamma, noise_std
        #demand_config: alpha_0, alpha_1, alpha_2, trend_eta
        #supply_config: beta
        #action_bounds: credit_delta_max, supply_push_max, friction_max
        #reward_params: imbalance_penalty_lambda, action_smoothness_mu
        #env_seed: Random seed for environment
        
        super().__init__()
        
        self.episode_length = time_config['episode_length']
        self.volatility_window = time_config['volatility_window']
        self.supply_lag = time_config['supply_lag']
        self.env_seed = env_seed
        
        self.max_credit_delta = action_bounds['credit_delta_max']
        self.max_supply_push = action_bounds['supply_push_max']
        self.max_friction = action_bounds['friction_max']
        
        self.dynamics = MarketDynamics(
            alpha_0=demand_config['alpha_0'],
            alpha_1=demand_config['alpha_1'],
            alpha_2=demand_config['alpha_2'],
            trend_eta=demand_config['trend_eta'],
            beta=supply_config['beta'],
            gamma=price_dynamics_config['gamma'],
            noise_std=price_dynamics_config['noise_std'],
            supply_lag=self.supply_lag
        )
        
        self.state_manager = StateManager(self.volatility_window)
        
        self.reward_fn = RewardFunction(
            lambda_imbalance=reward_params['imbalance_penalty_lambda'],
            mu_smoothness=reward_params['action_smoothness_mu']
        )
        
        self.rng = np.random.default_rng(self.env_seed)
        
        # State: [p_t, r_t, σ_t, z_t, c_t]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(5,),
            dtype=np.float32
        )
        
        # Action: [a_c, a_s, a_f]
        self.action_space = spaces.Box(
            low=np.array([-self.max_credit_delta, 0.0, 0.0], dtype=np.float32),
            high=np.array([self.max_credit_delta, self.max_supply_push, self.max_friction], dtype=np.float32),
            dtype=np.float32
        )
        
        self.log_price = 0.0
        self.log_return = 0.0
        self.credit_tightness = 0.0
        self.imbalance = 0.0
        self.supply_pipeline = 0.0
        self.pipeline_history = deque(maxlen=self.supply_lag + 1)
        
        self.current_step = 0

        
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        _ = seed
        _ = options
        
        self.rng = np.random.default_rng(self.env_seed)
        self.state_manager.reset()
        self.reward_fn.reset()
        
        self.log_price = 0.0
        self.log_return = 0.0
        self.credit_tightness = 0.0
        self.imbalance = 0.0
        self.supply_pipeline = 0.0
        
        self.pipeline_history.clear()
        for _ in range(self.supply_lag + 1):
            self.pipeline_history.append(0.0)
        
        self.current_step = 0
        
        obs = self._get_observation()
        info = {}
        
        return obs, info

    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert isinstance(self.action_space, gym.spaces.Box)
        action = np.clip(
            action,
            self.action_space.low,
            self.action_space.high
        )
        
        a_credit, a_supply, a_friction = action
        
        # 1. Compute expected return
        expected_return = self.dynamics.compute_expected_return(self.log_return)
        
        # 2. Compute demand
        demand = self.dynamics.compute_demand(expected_return, self.credit_tightness)
        
        # 3. Update supply pipeline
        self.supply_pipeline = self.dynamics.update_supply_pipeline(
            self.supply_pipeline,
            a_supply
        )
        self.pipeline_history.append(self.supply_pipeline)
        
        # 4. Compute actual supply from lagged pipeline
        pipeline_lagged = self.pipeline_history[0]  # k periods ago
        supply = self.dynamics.compute_supply(pipeline_lagged)
        
        # 5. Compute imbalance
        self.imbalance = self.dynamics.compute_imbalance(demand, supply)
        
        # 6. Apply market friction
        adjusted_imbalance = self.dynamics.apply_market_friction(self.imbalance, a_friction)
        
        # 7. Update price
        prev_log_price = self.log_price
        self.log_price = self.dynamics.compute_price_update(
            self.log_price,
            adjusted_imbalance,
            self.rng
        )
        
        # 8. Compute return
        self.log_return = self.log_price - prev_log_price
        
        # 9. Update credit
        self.credit_tightness = self.dynamics.compute_credit_update(
            self.credit_tightness,
            a_credit
        )
        
        # 10. Update state manager
        self.state_manager.add_return(self.log_return)
        
        # 11. Compute rolling volatility
        rolling_volatility = self.state_manager.compute_rolling_volatility()
        
        # 12. Compute reward
        reward = self.reward_fn.compute_reward(
            rolling_volatility,
            self.imbalance,
            action
        )
        
        # 13. Check termination
        self.current_step += 1
        terminated = self.current_step >= self.episode_length
        truncated = False
        
        # 14. Build observation
        obs = self._get_observation()
        
        # 15. Additional info
        info = {
            'log_price': self.log_price,
            'return': self.log_return,
            'rolling_volatility': rolling_volatility,
            'imbalance': self.imbalance,
            'credit_tightness': self.credit_tightness
        }
        
        return obs, reward, terminated, truncated, info

    
    def _get_observation(self) -> np.ndarray:
        rolling_volatility = self.state_manager.compute_rolling_volatility()
        
        obs = np.array([
            self.log_price,
            self.log_return,
            rolling_volatility,
            self.imbalance,
            self.credit_tightness
        ], dtype=np.float32)
        
        return obs
