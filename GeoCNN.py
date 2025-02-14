from GeoCNN.helpers import GeoCNN_helper
from dataclasses import dataclass, asdict, replace
from itertools import product
from typing import List, Dict
from GeoCNN.HyperparameterConfig import HyperparameterConfig

def args_generator() -> List[Dict]:
    
    """
    Generate all hyperparameter combinations using itertools.product.
    Returns:
        List[Dict]: A list of configurations as dictionaries.
    """
    
    # Base configuration
    base_config = HyperparameterConfig()

    # Parameter ranges for combinations
    param_ranges = {
        "DataSet": ["SWATplus_output"],
        "overwrite_training": [True],
        "target_array": ["perc"],
        "opt_lr": [5e-5],
        "weight_decay": [0.01],
        "embed_dim": [1024],
        "dropout": [0.40],
        "gpu_index": ["1,3"],
        "accumulation_steps": [1],
        "model":['VisionTransformerForRegression'],
        "batch_size": [24],
        "num_training_epochs": [1000],
        "early_stopping_patience": [50],
        "scheduler": ["CosineAnnealingHardRestarts"],
        "preloading" : [True],
    }
    
    # Generate all combinations
    keys, values = zip(*param_ranges.items())
    combinations = [dict(zip(keys, combo)) for combo in product(*values)]

    # Merge combinations with the base configuration
    args_list = [asdict(replace(base_config, **combo)) for combo in combinations]

    print(f"Total unique combinations: {len(args_list)}")

    return args_list

if __name__ == '__main__':

    args_list = args_generator()
    processes = []
    for args in args_list:
        GeoCNN_helper(args)