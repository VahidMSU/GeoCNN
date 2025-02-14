import h5py
import numpy as np
import os
from typing import Dict, Optional
from GeoCNN.full_inference import full_inference_helper
from GeoCNN.trainer import GeoClassCNN
from GeoCNN.utils import generate_model_name


def prepare_model_config(config: Dict, model_name: Optional[str] = None) -> Dict:
    """Prepare and validate model configuration."""
    model_name = model_name or generate_model_name(config)
    config.update({
        'model_name': model_name,
        'new_no_value': 1e-6
    })
    return config

def check_model_paths(model_name: str) -> Dict:
    """Generate standard paths for model artifacts."""
    base_path = f"/data/MyDataBase/out/VisionSystem/report/{model_name}"
    return {
        'base_path': base_path,
        'best_model_path': os.path.join(base_path, "best_model.pth"),
        'metrics_path': os.path.join(base_path, "metrics.txt")
    }

def handle_model_overwrite(paths: Dict, config: Dict) -> None:
    """Remove existing model directory if overwrite is enabled."""
    if config.get('overwrite_training', False) and os.path.exists(paths['base_path']):
        os.system(f"rm -r {paths['base_path']}")

def run_model_workflow(config: Dict, paths: Dict) -> Optional[Dict]:
    """Execute model training, prediction, and inference workflow."""
    geo_cnn = GeoClassCNN(config)
    
    if config.get('inference'):
        if not os.path.exists(paths['best_model_path']):
            geo_cnn.train_model()
            geo_cnn.predict()
        elif not os.path.exists(paths['metrics_path']):
            geo_cnn.predict()
        
        full_inference_helper(
            model_name=config['model_name'], 
            target_name=config['target_array'], 
            no_value=config['no_value'], 
            new_no_value=config['new_no_value']
        )
        return config
    
    if os.path.exists(paths['best_model_path']):
        return None

    geo_cnn.train_model()
    geo_cnn.predict()
    if config.get("DataSets") == "HydroGeoDataset":  
        
        full_inference_helper(
            model_name=config['model_name'], 
            target_name=config['target_array'], 
            no_value=config['no_value'], 
            new_no_value=config['new_no_value']
        )
    return config



def GeoCNN_helper(config: Dict, model_name: Optional[str] = None) -> Optional[Dict]:
    """Main function to manage model training and inference."""
    config = prepare_model_config(config, model_name)
    paths = check_model_paths(config['model_name'])
    
    handle_model_overwrite(paths, config)
    return run_model_workflow(config, paths)
