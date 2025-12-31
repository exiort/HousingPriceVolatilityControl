from __future__ import annotations
import pandas as pd
import os
from typing import List, Dict, Any



def write_training_summary(run_dir: str, training_data: List[Dict[str, Any]]) -> None:
    df = pd.DataFrame(training_data)
    
    columns = [
        'generation',
        'individual_id',
        'fitness',
        'learning_rate',
        'entropy_coefficient',
        'value_loss_coefficient',
        'clip_range',
        'imbalance_penalty_lambda',
        'action_smoothness_mu',
        'credit_delta_max',
        'supply_push_max',
        'friction_max'
    ]
    
    df = df[columns]
    
    output_path = os.path.join(run_dir, "training_summary.csv")
    df.to_csv(output_path, index=False)


def write_evaluation_summary(run_dir: str, eval_summary: List[Dict[str, Any]]) -> None:
    df = pd.DataFrame(eval_summary)
    
    df = df[['eval_seed', 'mean_rolling_vol']]
    
    output_path = os.path.join(run_dir, "evaluation_summary.csv")
    df.to_csv(output_path, index=False)


def write_train_timeseries(run_dir: str, best_individual_data: Dict[str, Any]) -> None:
    if not best_individual_data:
        df = pd.DataFrame(columns=pd.Index([
            'episode', 'timestep', 'log_price', 'return', 'rolling_volatility',
            'imbalance_z', 'credit_tightness', 'action_credit_delta',
            'action_supply_push', 'action_friction', 'reward'
        ]))
    else:
        trajectory = best_individual_data.get('trajectory', [])
        
        rows = []
        for timestep, step_data in enumerate(trajectory):
            row = {
                'episode': 0,
                'timestep': timestep,
                'log_price': step_data['log_price'],
                'return': step_data['return'],
                'rolling_volatility': step_data['rolling_volatility'],
                'imbalance_z': step_data['imbalance'],
                'credit_tightness': step_data['credit_tightness'],
                'action_credit_delta': step_data['action_credit_delta'],
                'action_supply_push': step_data['action_supply_push'],
                'action_friction': step_data['action_friction'],
                'reward': step_data['reward']
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
    
    
    output_path = os.path.join(run_dir, "train_timeseries.csv")
    df.to_csv(output_path, index=False)


def write_eval_timeseries(run_dir: str, eval_timeseries: List[Dict[str, Any]]) -> None:
    for entry in eval_timeseries:
        if 'imbalance' in entry:
            entry['imbalance_z'] = entry.pop('imbalance')
    
    df = pd.DataFrame(eval_timeseries)
    
    columns = [
        'eval_seed', 'episode', 'timestep', 'log_price', 'return',
        'rolling_volatility', 'imbalance_z', 'credit_tightness',
        'action_credit_delta', 'action_supply_push', 'action_friction', 'reward'
    ]
    
    df = df[columns]
    
    output_path = os.path.join(run_dir, "eval_timeseries.csv")
    df.to_csv(output_path, index=False)
