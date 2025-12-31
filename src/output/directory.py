from __future__ import annotations
import os
import json
from datetime import datetime
from typing import Dict, Any

from src.utils.seeding import SeedManager



def create_run_directory() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dirname = f"run_{timestamp}"
    
    os.makedirs(dirname, exist_ok=True)
    
    return dirname


def save_config_copy(run_dir: str, config: Dict[str, Any]) -> None:
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def save_seeds(run_dir: str, seed_manager:SeedManager) -> None:
    seeds_path = os.path.join(run_dir, "seeds.json")
    seed_manager.save_to_file(seeds_path)
