# HousingPriceVolatilityControl

A hybrid Evolutionary Computation (EC) and Reinforcement Learning (RL) framework designed to control and mitigate housing price volatility. Rather than attempting to forecast prices, this project formulates housing market regulation as a continuous control problem. An RL agent indirectly influences price dynamics within a stochastic environment through macroeconomic levers: credit conditions, supply incentives, and market frictions.

## About The Project

This repository provides a controlled simulation environment that captures essential demand-supply imbalances and delayed policy effects in housing markets. By jointly optimizing policy parameters using policy gradient methods (PPO) and evolutionary search, the framework discovers robust policy configurations capable of minimizing rolling price volatility without relying on direct price-setting interventions.

## Features

* **Hybrid Optimization Engine:** Combines Reinforcement Learning (Proximal Policy Optimization via `stable-baselines3`) with an Evolutionary Algorithm to evolve hyperparameters, reward structures, and action bounds simultaneously.
* **Stochastic Housing Environment:** A custom `gymnasium` environment simulating housing market mechanics, including delayed supply responses, demand trends, and structural market noise.
* **Multi-Channel Policy Control:** The agent regulates the market using three distinct continuous action channels:
  * *Credit Conditions:* Adjusting borrowing ease.
  * *Supply Incentives:* Pushing housing supply allocations.
  * *Market Frictions:* Imposing transaction costs or regulatory drag.
* **Configuration-Driven:** All market dynamics, RL bounds, evolutionary parameters, and random seeds are fully specified via JSON configuration files to ensure exact reproducibility across experiments.

## Project Structure

* `main.py`: Entry point for executing the training and evaluation pipeline.
* `configs/`: Contains JSON configuration files (e.g., `base_config.json`, `s1_high.json`, `s2_aggresive.json`).
* `src/`
  * `environment/`: Core simulation logic (`housing_env.py`, `dynamics.py`, `state.py`, `reward.py`).
  * `training/`: RL setup, fitness evaluation, and the PPO trainer.
  * `evolution/`: Genetic algorithm components (`population.py`, `individual.py`, `genetic_ops.py`).
  * `evaluation/`: Post-training policy evaluation modules.
  * `output/`: Handlers for saving experiment metrics and CSV data.
  * `utils/`: Configuration parsers and global seeding utilities.

## Installation

1. Ensure you have Python 3.8+ installed.
2. Clone the repository.
3. Install the required dependencies:

    pip install -r requirements.txt

    **Dependencies Include:**
    * `numpy`, `scipy`, `pandas` (Numerical and data operations)
    * `gymnasium` (Environment API)
    * `stable-baselines3` (Reinforcement learning algorithms)
    * `tqdm` (Progress tracking)

## Usage

To run the training and evolution pipeline, you **must** provide a configuration file. Execute the `main.py` script and pass the path to your desired JSON configuration file as an argument:

    python main.py configs/base_config.json

Upon execution, the system will initialize the population, run the evolutionary RL pipeline for the specified number of generations, and output performance metrics (like fitness, rolling volatility, and policy behaviors) to the designated output directories.

## Configuration Parameters Guide

All evolutionary runs and environment setups are strictly controlled by a configuration file. Below is an exhaustive breakdown of all parameters required in the configuration, their scopes, and examples (based on `base_config.json`).

### 1. `time` (Object)
Controls the temporal structure and memory of the simulation environment.
* `episode_length` (Integer): The total number of timesteps per simulation episode. *Example: 240*
* `volatility_window` (Integer): The rolling window size used to calculate the target price volatility. *Example: 6*
* `supply_lag` (Integer): The delay (in timesteps) between a supply incentive action and actual market delivery. *Example: 3*

### 2. `price_dynamics` (Object)
Defines the inherent mathematical behavior of the market prices.
* `gamma` (Float): The mean-reversion or base structural drift coefficient for prices. *Example: 0.2*
* `noise_std` (Float): Standard deviation of the exogenous stochastic shocks applied to the market. *Example: 0.02*

### 3. `demand` (Object)
Parameters defining how housing demand reacts to market state and agent actions.
* `alpha_0` (Float): Baseline intrinsic demand. *Example: 0.0*
* `alpha_1` (Float): Demand sensitivity to credit conditions. *Example: 0.5*
* `alpha_2` (Float): Demand sensitivity to market frictions. *Example: 1.0*
* `trend_eta` (Float): The momentum or persistence of demand trends over time. *Example: 0.8*

### 4. `supply` (Object)
* `beta` (Float): The baseline sensitivity of the market to new housing supply. *Example: 0.3*

### 5. `training` (Object)
* `timesteps` (Integer): The total number of PPO environment interaction steps allocated per individual in a generation. *Example: 30000*

### 6. `ppo_param_bounds` (Object)
Defines the minimum and maximum boundaries for the evolutionary algorithm to search for optimal PPO hyperparameters. Each sub-object requires a `min` and `max` float value.
* `learning_rate`: Step size for the optimizer. *Example: {"min": 0.00005, "max": 0.0002}*
* `entropy_coefficient`: Promotes exploration by penalizing deterministic policies. *Example: {"min": 0.005, "max": 0.03}*
* `value_loss_coefficient`: The weight of the value function loss in the total objective. *Example: {"min": 0.4, "max": 0.8}*
* `clip_range`: The PPO clipping parameter limiting policy updates. *Example: {"min": 0.15, "max": 0.25}*

### 7. `reward_param_bounds` (Object)
Defines the search space for evolving the agent's reward function components.
* `imbalance_penalty_lambda`: Penalty weight applied when supply and demand are heavily misaligned. *Example: {"min": 0.05, "max": 0.3}*
* `action_smoothness_mu`: Penalty weight penalizing high-frequency or erratic policy adjustments (encourages smooth control). *Example: {"min": 0.02, "max": 0.12}*

### 8. `action_bound_bounds` (Object)
Sets the evolutionary search space for the absolute maximum strength of the agent's interventions.
* `credit_delta_max`: Maximum permissible change to credit conditions per step. *Example: {"min": 0.08, "max": 0.2}*
* `supply_push_max`: Maximum allowable supply incentive intervention. *Example: {"min": 0.3, "max": 0.8}*
* `friction_max`: Maximum level of market friction the agent can apply. *Example: {"min": 0.15, "max": 0.45}*

### 9. `evolution` (Object)
Configures the Genetic Algorithm responsible for outer-loop optimization.
* `population_size` (Integer): Number of RL agents in a single generation. *Example: 12*
* `elite_fraction` (Float): The percentage of top-performing individuals automatically carried over to the next generation. *Example: 0.2*
* `tournament_size` (Integer): Number of individuals randomly selected for tournament mating selection. *Example: 3*
* `mutation_std` (Object): The standard deviation for Gaussian mutations applied to an offspring's genome. Requires keys matching the evolved parameters (e.g., `learning_rate`, `entropy_coefficient`, `action_smoothness_mu`, `supply_push_max`). *Example: {"learning_rate": 0.3, "friction_max": 0.04, ...}*

### 10. `seeds` (Object)
Ensures total experimental reproducibility.
* `env_train_seed` (Integer): Random seed used for the environment during the training phase. *Example: 2001*
* `env_eval_seeds` (Array of Integers): A list of different seeds used to evaluate the robustness of a trained policy across unseen market trajectories. *Example: [213, 215, 482, 567, 568]*
* `rl_seed` (Integer): Seed for PPO network initialization and action sampling. *Example: 1920*
* `ec_seed` (Integer): Seed governing the genetic algorithm's crossover and mutation operations. *Example: 1923*

---

## 📜 License

This project is licensed under the MIT License. Copyright (c) 2026 Buğrahan İmal. You are free to use, copy, modify, merge, publish, and distribute this software as per the license conditions.
