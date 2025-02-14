from typing import Optional
from GeoCNN.full_inference import full_inference_helper
from GeoCNN.trainer import GeoClassCNN
from GeoCNN.utils import generate_model_name
from dataclasses import asdict
from GeoCNN.HyperparameterConfig import HyperparameterConfig
import os

def prepare_prediction_config(model_name: str) -> HyperparameterConfig:
    """
    Prepare and validate the configuration for prediction.
    """
    base_path = f"/data/MyDataBase/out/VisionSystem/report/{model_name}"
    config = HyperparameterConfig(
        inference=True,
        model_name=model_name,
        output_path="/data/MyDataBase/out/VisionSystem/report",
        batch_size=25,
        target_array="ET",
        seq_len=80,
        DataSet="HydroGeoDataset",
        gpu_index="0",
        hydrogeodataset_path="/data/MyDataBase/HydroGeoDataset_ML_250.h5",
        swatplus_output_path =  "/data/MyDataBase/out/SWATplus_output/CentralSWAT_data.h5", 
        extracted_dataset_path="ml_data/GeoCNN_data.h5",
        best_model_path=f"{base_path}/best_model.pth",
        metrics_path=f"{base_path}/metrics.txt",
    )
    assert os.path.exists(config.best_model_path), f"Best model not found at {config.best_model_path}"
    return config


def predict_only(model_name: str) -> None:
    """
    Helper function to load the model and run predictions only.
    """
    # Prepare the configuration using HyperparameterConfig
    config = prepare_prediction_config(model_name)

    # Convert dataclass to dictionary for GeoClassCNN
    config_dict = asdict(config)

    # Initialize the model trainer with the prediction configuration
    geo_cnn = GeoClassCNN(config_dict)

    print(f"Running predictions for model: {model_name}")

    # Run prediction
    geo_cnn.predict()

    # Optionally run full inference helper
    #full_inference_helper(
    #    model_name=config.model_name,
    #   target_name=config.target_array,

    #)

if __name__ == "__main__":
    model_name = "ET_CNNTransformerRegressor_v8_bs28_bw64_lr0_0001_wd0_01_epochs1000_nh8_nl6_fe4_embs1024_dropout0_39"  # Replace with the desired model name    
    predict_only(model_name)