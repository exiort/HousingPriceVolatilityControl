# Housing Volatility Control (RL + EC)

A reinforcement learning (RL) + evolutionary computation (EC) system for controlling housing market price volatility.

## Overview

This project implements a control system that uses PPO (Proximal Policy Optimization) to learn policies for minimizing rolling house price volatility. The system evolves hyperparameters, reward coefficients, and action bounds using a genetic algorithm to optimize performance.

**Key Features:**
- 5D continuous state space (log price, return, rolling volatility, imbalance, credit tightness)
- 3D continuous action space (credit policy, supply push, market friction)
- Fixed MLP architecture (2×64, tanh) with evolved PPO hyperparameters
- (μ+λ) genetic algorithm with elitism and tournament selection
- Reproducible multi-seed evaluation


## Usage

### Running a Simulation

```bash
python main.py <config_path>
```

**Example:**
```bash
python main.py my_config.json
```

### Configuration File

The simulation requires a JSON configuration file. See the specification for the complete config schema.

**Minimal example:**
```json
{
  "time": {
    "episode_length": 240,
    "volatility_window": 6,
    "supply_lag": 3
  },
  "price_dynamics": {
    "gamma": 0.2,
    "noise_std": 0.02
  },
  "demand": {
    "alpha_0": 0.0,
    "alpha_1": 0.5,
    "alpha_2": 1.0,
    "trend_eta": 0.8
  },
  "supply": {
    "beta": 0.3
  },
  "training": {
    "timesteps": 200000
  },
  "ppo_param_bounds": { ... },
  "reward_param_bounds": { ... },
  "action_bound_bounds": { ... },
  "evolution": { ... },
  "seeds": {
    "env_train_seed": 42,
    "env_eval_seeds": [101, 202, 303],
    "rl_seed": 1234,
    "ec_seed": 5678
  }
}
```

## Output Structure

Each simulation creates a timestamped directory:

```
run_YYYYMMDD_HHMMSS/
├── config.json              # Copy of input configuration
├── seeds.json               # Seed record for reproducibility
├── training_summary.csv     # All individuals and their fitness
├── evaluation_summary.csv   # Mean volatility per eval seed
├── train_timeseries.csv     # Best individual trajectory on train seed
└── eval_timeseries.csv      # Trajectories on all eval seeds
```

### CSV Schemas

**training_summary.csv:**
```
generation,individual_id,fitness,learning_rate,entropy_coefficient,
value_loss_coefficient,clip_range,imbalance_penalty_lambda,
action_smoothness_mu,max_credit_delta,max_supply_push,max_friction
```

**evaluation_summary.csv:**
```
eval_seed,mean_rolling_vol
```

**train_timeseries.csv:**
```
episode,timestep,log_price,return,rolling_volatility,imbalance_z,
credit_tightness,action_credit_delta,action_supply_push,action_friction,reward
```

**eval_timeseries.csv:**
```
eval_seed,episode,timestep,log_price,return,rolling_volatility,imbalance_z,
credit_tightness,action_credit_delta,action_supply_push,action_friction,reward
```

## Project Structure

```
src/
├── environment/        # Housing market environment
│   ├── housing_env.py  # Main Gymnasium environment
│   ├── dynamics.py     # Market dynamics (demand, supply, price)
│   ├── state.py        # State tracking and volatility
│   └── reward.py       # Reward function
├── evolution/          # Genetic algorithm
│   ├── individual.py   # Parameter representation
│   ├── genetic_ops.py  # Selection and mutation
│   └── population.py   # Population management
├── training/           # RL + EC training
│   ├── ppo_trainer.py  # PPO wrapper
│   ├── fitness.py      # Fitness evaluation
│   └── pipeline.py     # RL + EC integration
├── evaluation/         # Multi-seed evaluation
│   └── evaluator.py    # Final evaluation
├── output/             # Output management
│   ├── directory.py    # Run directory creation
│   └── csv_writer.py   # CSV output writers
└── utils/              # Utilities
    ├── config.py       # Configuration loading
    └── seeding.py      # Seed management
```

## Seed Policy

The system uses **4 independent seeds** for complete reproducibility:

1. **env_train_seed**: Fixed environment seed for all training
2. **env_eval_seeds**: List of seeds for final evaluation only
3. **rl_seed**: PPO algorithm seed
4. **ec_seed**: Genetic algorithm seed

**Critical:** Training ALWAYS uses `env_train_seed`. Evaluation seeds are used ONLY for final multi-seed testing.

## Mathematical Framework

### Environment Dynamics

**Demand:**
```
D_t = α₀ + α₁·E[r_t] − α₂·c_t
where E[r_t] = η·r_{t-1}
```

**Supply:**
```
π_t = π_{t-1} + a_s  (pipeline)
S_t = β·π_{t-k}      (actual supply, k periods lag)
```

**Imbalance & Price:**
```
z_t = D_t − S_t
ẑ_t = z_t·(1 − a_f)  (friction-adjusted)
p_{t+1} = p_t + γ·ẑ_t + ε_t
```

**Reward:**
```
R_t = −σ_t − λ·|z_t| − μ·||a_t − a_{t-1}||₂
```

### Fitness Metric

Fitness = Mean rolling volatility (lower is better)

```
J = (1/T)·Σ σ_t
```

## Design Principles

1. **Control, not prediction**: Agent controls policies, not prices directly
2. **RL + EC always together**: Hyperparameters evolved during training
3. **Simulation-analysis separation**: This project only generates data
4. **Strict reproducibility**: 4 independent seeds, fixed architecture
5. **No interpretation**: Direct implementation of specification

## Notes

- This is a simulation-only project. Analysis tools are separate.
- The evolved parameters are specific to the training seed and configuration.
- Episode length and volatility window affect rolling volatility calculations.
