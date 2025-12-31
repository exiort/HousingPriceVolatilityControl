from __future__ import annotations
import argparse
import sys

from src.utils.config import load_config
from src.utils.seeding import SeedManager
from src.training.pipeline import TrainingPipeline
from src.evaluation.evaluator import Evaluator
from src.output.directory import create_run_directory, save_config_copy, save_seeds
from src.output.csv_writer import (
    write_training_summary,
    write_evaluation_summary,
    write_train_timeseries,
    write_eval_timeseries
)



def main():
    parser = argparse.ArgumentParser(
        description='Housing Volatility Control - RL + EC Simulation'
    )
    parser.add_argument(
        'config_path',
        type=str,
        help='Path to JSON configuration file'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config_path}")
    try:
        config = load_config(args.config_path)
    except Exception as e:
        print(f"ERROR: Failed to load configuration: {e}")
        sys.exit(1)
    
    # Initialize seed manager
    seed_manager = SeedManager(config['seeds'])
    
    print("\n" + "="*60)
    print("HOUSING VOLATILITY CONTROL - RL + EC SIMULATION")
    print("="*60)
    print(f"\nSeeds:")
    print(f"  Environment (train): {seed_manager.get_env_train_seed()}")
    print(f"  Environment (eval):  {seed_manager.get_env_eval_seeds()}")
    print(f"  RL algorithm:        {seed_manager.get_rl_seed()}")
    print(f"  Evolution:           {seed_manager.get_ec_seed()}")
    
    print(f"\nCreating run directory...")
    run_dir = create_run_directory()
    print(f"  Output directory: {run_dir}")
    

    save_config_copy(run_dir, config)
    save_seeds(run_dir, seed_manager)
    
    num_generations = config.get('num_generations', 10)
    
    print(f"\nTraining configuration:")
    print(f"  Population size:     {config['evolution']['population_size']}")
    print(f"  Generations:         {num_generations}")
    print(f"  PPO timesteps:       {config['training']['timesteps']}")
    print(f"  Episode length:      {config['time']['episode_length']}")
    
    # Initialize and run training pipeline
    print(f"\n{'='*60}")
    print("TRAINING PHASE")
    print("="*60)
    
    pipeline = TrainingPipeline(config, seed_manager)
    best_individual, training_data = pipeline.run_evolution(num_generations)
    
    print(f"\nTraining complete!")
    print(f"  Best fitness (mean rolling volatility): {best_individual.fitness:.6f}")
    print(f"\nBest individual parameters:")
    for param_name, param_value in best_individual.get_all_params().items():
        print(f"  {param_name}: {param_value:.6f}")
    
    # Save training outputs
    print(f"\nSaving training outputs...")
    write_training_summary(run_dir, training_data)
    
    print(f"\nGetting best individual trajectory on training seed...")
    from src.training.fitness import evaluate_individual
    _, best_trajectory = evaluate_individual(
        individual=best_individual,
        config=config,
        env_seed=seed_manager.get_env_train_seed(),
        rl_seed=seed_manager.get_rl_seed(),
        num_eval_episodes=1  # Just one episode for trajectory
    )
    
    write_train_timeseries(run_dir, {'trajectory': best_trajectory})
    
    # Evaluation phase
    print(f"\n{'='*60}")
    print("EVALUATION PHASE")
    print("="*60)
    
    evaluator = Evaluator(config, seed_manager)
    eval_summary, eval_timeseries = evaluator.evaluate_best_individual(
        best_individual,
        num_episodes_per_seed=5
    )
    
    print(f"\nSaving evaluation outputs...")
    write_evaluation_summary(run_dir, eval_summary)
    write_eval_timeseries(run_dir, eval_timeseries)
    
    print(f"\n{'='*60}")
    print("SIMULATION COMPLETE")
    print("="*60)
    print(f"\nOutputs saved to: {run_dir}/")
    print(f"  - config.json")
    print(f"  - seeds.json")
    print(f"  - training_summary.csv")
    print(f"  - evaluation_summary.csv")
    print(f"  - train_timeseries.csv")
    print(f"  - eval_timeseries.csv")
    
    print(f"\nEvaluation results:")
    for entry in eval_summary:
        print(f"  Seed {entry['eval_seed']}: {entry['mean_rolling_vol']:.6f}")
    
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
