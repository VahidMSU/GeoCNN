from dataclasses import dataclass
from typing import List, Dict
from dataclasses import field



@dataclass
class HyperparameterConfig:
    report_path: str = None
    model_name: str = None
    best_model_path: str = None
    metrics_path: str = None
    # General settings
    plotting: bool = False
    load_tensors: bool = False
    overwrite_training: bool = False
    overwrite: bool = False
    inference: bool = False
    initial_condition: bool = False
    
    # Data distribution and training settings
    gpu_index: str = "0"
    seq_len: int = 80
    accumulation_steps: int = 1
    preloaded: bool = True

    # Scheduler settings
    scheduler: str = "CosineAnnealingHardRestarts"
    scheduler_lib: str = "transformers"
    num_training_epochs: int = 300
    data_pipeline_queue_size: int = 10

    ## Early stopping settings
    overfitting_window: int = 5
    check_overfitting_after: int = 500  ## inactive overfitting check by default
    overfit_threshold: float = 0.01
    num_warmup_steps: int = 10         ### excluding later. its now calculated by the percentage of the total number of steps
    early_stopping_patience: int = 30  
    # Data settings
    DataSet: str = None # "SWATplus_output", "HydroGeoDataset"
    no_value_distribution: str = "clean_training"
      
    no_value: float = -999 
    new_no_value: float = -1e-6
    hydrogeodataset_path: str = "/data/MyDataBase/HydroGeoDataset_ML_250.h5"
    extracted_dataset_path: str = "ml_data/GeoCNN_data.h5"
    RESOLUTION: int = 250
    target_array: str = None
    output_path: str = "/data/MyDataBase/out/VisionSystem/report"
    start_year: int = 2002
    end_year: int = 2021

    # Model settings
    model: str = "CNNTransformerRegressor_v8"
    num_classes: int = 1
    embed_dim: int = 1024
    num_heads: int = 8
    num_layers: int = 6
    forward_expansion: int = 4
    dropout: float = 0.5

    # 'VisionTransformer' or 'VisionTransformerForRegression'
    mlp_dim: int = 512
    depth: int = 6
    
    # Optimization settings
    optimizer: str = "AdaBelief"
    opt_lr: float = 5e-5
    weight_decay: float = 0.001
    gradient_cliping: bool = True
    loss_function: str = "SpatioTemporalLoss"
    max_norm: float = 1.0
    num_cycles: int = 1

    # Batch and window settings
    batch_size: int = 12
    batch_window: int = 64
    preloading: bool = False
    
    ### SWATplus model settings
    swatplus_output_path: str = "/data/MyDataBase/out/SWATplus_output/CentralSWAT_data.h5"
    swat_target_variable: str = "wateryld"
    swat_dynamic_variables: List[str] = field(default_factory=lambda: ['et', 'precip', 'snofall', "wateryld"])
    swat_static_variables: Dict[str, List[str]] = field(default_factory=lambda: {
        "Soil": [
            'alb_30m', 'awc_30m', 'bd_30m', 'caco3_30m', 'carbon_30m', 'clay_30m',
            'dp_30m', 'ec_30m', 'ph_30m', 'rock_30m', 'silt_30m', 'soil_30m', 'soil_k_30m'
        ],
        "DEM": ['dem', 'demslp']
    })