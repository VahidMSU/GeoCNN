import os
import torch
import logging
import numpy as np
import imageio
from PIL import Image  # Required for high DPI image resizing
import matplotlib.pyplot as plt
from typing import Any, Dict, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch
from typing import Tuple

def calculate_metrics(target_array, predictions_array):
    def nse(obs, sim):
        denominator = np.sum((obs - np.mean(obs)) ** 2)
        return (
            np.nan
            if denominator == 0
            else 1 - np.sum((obs - sim) ** 2) / denominator
        )

    def mse(obs, sim):
        return np.mean((obs - sim) ** 2)

    def rmse(obs, sim):
        return np.sqrt(np.mean((obs - sim) ** 2))

    nse_val = nse(target_array.flatten(), predictions_array.flatten())
    mse_val = mse(target_array.flatten(), predictions_array.flatten())
    rmse_val = rmse(target_array.flatten(), predictions_array.flatten())
    return nse_val, mse_val, rmse_val


def test_forward_pass(
    #inputs: torch.Tensor, 
    dynamic_inputs: torch.Tensor,
    static_inputs: torch.Tensor,
    categorical_inputs: torch.Tensor,
    targets: torch.Tensor, 
    config: Dict[str, Any], 
    model: nn.Module, 
    device: torch.device, 
    no_value: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform forward pass with flexible handling of model outputs and targets.

    Args:
        inputs: Input tensor
        targets: Target tensor
        config: Configuration dictionary
        model: Neural network model
        device: Computation device
        no_value: Value representing invalid/missing data

    Returns:
        Tuple of validated targets and model predictions
    """
    # Move inputs and targets to specified device
    #inputs, targets = inputs.to(device), targets.to(device)
    dynamic_inputs, static_inputs, categorical_inputs, targets = dynamic_inputs.to(device), static_inputs.to(device), categorical_inputs.to(device), targets.to(device)

    with torch.no_grad():
        # Handle initial condition if specified
        preds = (
            model(dynamic_inputs, static_inputs, categorical_inputs, targets[:, 0:1]) if config.get('initial_condition', False)
            else model(dynamic_inputs, static_inputs, categorical_inputs)
        )

        # Normalize prediction dimensions
        if preds.ndim == 4:  # Missing time dimension
            preds = preds.unsqueeze(1)
        elif preds.shape[1] != targets.shape[1]:
            preds = preds.repeat(1, targets.shape[1], 1, 1, 1)

        # Create mask for valid values
        mask = (targets != no_value).float()

        # Handle classification vs regression
        if config.get('num_classes', 1) > 1:
            valid_targets = targets.long() * mask.long()
        else:
            valid_targets = targets * mask

    return valid_targets, preds

def run_prediction_evaluation(
    config: Dict[str, Any], 
    model: nn.Module, 
    data_batch: DataLoader, 
    report_path: str, 
    no_value: float, 
    device: torch.device,
    step: str
) -> Dict[str, float]:
    """
    Evaluate model performance on test batches.

    Args:
        config: Model configuration
        model: Trained neural network model
        data_batch: DataLoader containing test data
        report_path: Directory to store evaluation results
        no_value: Value representing invalid/missing data
        device: Computation device

    Returns:
        Dictionary of average performance metrics
    """
    metrics = {
        'nse': [],
        'mse': [],
        'rmse': []
    }

    for batch_idx, (dynamic_inputs, static_inputs, categorical_inputs, targets) in enumerate(data_batch):
        valid_targets, preds = test_forward_pass(
           dynamic_inputs, static_inputs, categorical_inputs , targets, config, model, device, no_value
        )
        #nse, mse, rmse = generate_test_gif(
        #    valid_targets, preds, report_path, batch_idx, step, no_value
        #)

        nse, mse, rmse = generate_test_mp4(
            valid_targets, preds, report_path, batch_idx, step, no_value
        )
        
        rmse_correlation_figs(
            valid_targets, preds, batch_idx, step, report_path, no_value 
        )

        metrics['nse'].append(nse)
        metrics['mse'].append(mse)
        metrics['rmse'].append(rmse)

    # Compute and store average metrics
    avg_metrics = {
        key: np.nanmean(values) 
        for key, values in metrics.items()
    }

    store_metrics(report_path, 
        metrics['nse'], 
        metrics['mse'], 
        metrics['rmse'],
        step
    )

    return avg_metrics


def store_metrics(report_path, nse_vals, mse_vals, rmse_vals, step):
    """
    Store the metrics in a text file.
    """

    #### clamp the values to 97.5 nanpercentile
    nse_vals = np.clip(nse_vals, a_min=None, a_max=np.nanpercentile(nse_vals, 97.5))
    mse_vals = np.clip(mse_vals, a_min=None, a_max=np.nanpercentile(mse_vals, 97.5))
    rmse_vals = np.clip(rmse_vals, a_min=None, a_max=np.nanpercentile(rmse_vals, 97.5))

    ### clamp the values to 2.5 nanpercentile

    nse_vals = np.clip(nse_vals, a_min=np.nanpercentile(nse_vals, 2.5), a_max=None)
    mse_vals = np.clip(mse_vals, a_min=np.nanpercentile(mse_vals, 2.5), a_max=None)
    rmse_vals = np.clip(rmse_vals, a_min=np.nanpercentile(rmse_vals, 2.5), a_max=None)

    log_results(report_path, nse_vals, mse_vals, rmse_vals, step)

def log_results(report_path, nse_vals, mse_vals, rmse_vals, step):
    ### setup logger
    from GeoCNN.utils import LoggerSetup
    logger = LoggerSetup(report_path)
    logger.setup_logger('metrics')

    logger.info(f"Metrics for the model, step: {step}")
    logger.info("")
    logger.info("Average Metrics")
    logger.info(f"Step: {step} Average NSE: {np.nanmean(nse_vals):.4f}, Median NSE: {np.nanmedian(nse_vals):.4f}, Max NSE: {np.nanmax(nse_vals):.4f}")
    logger.info(f"Step: {step} Average MSE: {np.nanmean(mse_vals):.4f}, Median MSE: {np.nanmedian(mse_vals):.4f}, Max MSE: {np.nanmax(mse_vals):.4f}")
    logger.info(f"Step: {step} Average RMSE: {np.nanmean(rmse_vals):.4f}, Median RMSE: {np.nanmedian(rmse_vals):.4f}, Max RMSE: {np.nanmax(rmse_vals):.4f}")

    logger.info("")
    logger.info("Metrics for each batch")
    for i in range(len(nse_vals)):
        logger.info(f"step: {step} Batch {i + 1} - NSE: {nse_vals[i]:.4f}, MSE: {mse_vals[i]:.4f}, RMSE: {rmse_vals[i]:.4f}")
    logger.info("")
    logger.info("End of file")
        
def generate_test_mp4(
    valid_targets: torch.Tensor,
    preds: torch.Tensor,
    report_path: str,
    batch_idx: int,
    step: str,
    no_value: float = -999,
    fps: int = 2,
    resolution: Tuple[int, int] = (1024, 512)
) -> Tuple[float, float, float]:
    import imageio
    import cv2  # OpenCV for video writing
    """
    Generate an MP4 video with ground truth, predictions, and absolute residual error.
    Compute performance metrics.

    Args:
        valid_targets: Ground truth tensor.
        preds: Predicted tensor.
        report_path: Directory to save results.
        batch_idx: Batch index for naming.
        step: Identifier for the current step.
        no_value: Value representing invalid data.
        fps: Frames per second for the MP4 video.
        resolution: Resolution of the output frames (width, height).

    Returns:
        Tuple of mean NSE, MSE, and RMSE.
    """
    video_name = f"predictions_vs_ground_truth_batch_{batch_idx}_{step}.mp4"
    time_steps = min(60, valid_targets.shape[1])
    
    metrics = {
        'nse': [],
        'mse': [],
        'rmse': []
    }

    # Prepare data
    preds_np = preds[:, :time_steps, 0, :, :].cpu().detach().numpy()
    targets_np = valid_targets[:, :time_steps, 0, :, :].cpu().detach().numpy()

    mask = (targets_np != no_value)
    preds_np[~mask] = np.nan
    targets_np[~mask] = np.nan

    # Compute global color scale range
    combined_data = np.concatenate([preds_np[mask], targets_np[mask]])
    global_min = np.nanpercentile(combined_data, 1)
    global_max = np.nanpercentile(combined_data, 99)
    residual_max = global_max - global_min  # Set scale for residual error

    # Create directory for video output
    os.makedirs(report_path, exist_ok=True)
    video_path = os.path.join(report_path, video_name)

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, resolution)

    for t in range(time_steps):
        preds_t = preds_np[:, t, :, :].squeeze()
        targets_t = targets_np[:, t, :, :].squeeze()

        # Skip frames with no valid data
        if np.all(np.isnan(targets_t)) or np.all(np.isnan(preds_t)):
            continue

        # Compute absolute residual error
        residual_error = np.abs(targets_t - preds_t)

        # Compute metrics
        nse_val, mse_val, rmse_val = calculate_metrics(targets_t, preds_t)
        metrics['nse'].append(nse_val)
        metrics['mse'].append(mse_val)
        metrics['rmse'].append(rmse_val)

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        im1 = axes[0].imshow(targets_t, cmap='viridis', interpolation='nearest', vmin=global_min, vmax=global_max)
        axes[0].set_title(f"Ground Truth (t={t})")
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

        im2 = axes[1].imshow(preds_t, cmap='viridis', interpolation='nearest', vmin=global_min, vmax=global_max)
        axes[1].set_title(f"Prediction (t={t})")
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

        im3 = axes[2].imshow(residual_error, cmap='viridis', interpolation='nearest', vmin=0, vmax=residual_max)
        axes[2].set_title(f"Absolute Residual Error (t={t})")
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

        # Convert plot to image
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # Resize frame to target resolution
        frame_resized = cv2.resize(frame, resolution, interpolation=cv2.INTER_LINEAR)
        video_writer.write(cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR))

        plt.close(fig)

    video_writer.release()
    print(f"MP4 video saved successfully to {video_path}")

    # Compute and return mean metrics
    return (
        np.nanmean(metrics['nse']),
        np.nanmean(metrics['mse']),
        np.nanmean(metrics['rmse'])
    )


def generate_test_gif(
    valid_targets: torch.Tensor,
    preds: torch.Tensor,
    report_path: str,
    batch_idx: int,
    step: str,  
    no_value: float = -999
) -> Tuple[float, float, float]:
    """
    Generate visualization GIF with ground truth, predictions, and absolute residual error.
    Compute performance metrics.

    Args:
        valid_targets: Ground truth tensor
        preds: Predicted tensor
        report_path: Directory to save results
        batch_idx: Batch index for naming
        no_value: Value representing invalid data

    Returns:
        Tuple of mean NSE, MSE, and RMSE
    """
    gif_name = f"predictions_vs_ground_truth_batch_{batch_idx}_{step}.gif"
    time_steps = min(60, valid_targets.shape[1])
    target_dpi_resolution = (1024, 512)

    metrics = {
        'nse': [],
        'mse': [],
        'rmse': []
    }
    frames = []

    # Compute the global color scale range
    preds_np = preds[:, :time_steps, 0, :, :].cpu().detach().numpy()
    targets_np = valid_targets[:, :time_steps, 0, :, :].cpu().detach().numpy()

    mask = (targets_np != no_value)
    preds_np[~mask] = np.nan
    targets_np[~mask] = np.nan

    # Compute 5th and 95th percentiles for color scaling
    combined_data = np.concatenate([preds_np[mask], targets_np[mask]])
    global_min = np.nanpercentile(combined_data, 1)
    global_max = np.nanpercentile(combined_data, 99)
    
    residual_max = global_max - global_min  # Set scale for residual error

    for t in range(time_steps):
        # Extract and process current time step
        preds_t = preds_np[:, t, :, :].squeeze()
        targets_t = targets_np[:, t, :, :].squeeze()

        # Skip frames with no valid data
        if np.all(np.isnan(targets_t)) or np.all(np.isnan(preds_t)):
            continue

        # Compute absolute residual error
        residual_error = np.abs(targets_t - preds_t)

        # Compute metrics
        nse_val, mse_val, rmse_val = calculate_metrics(targets_t, preds_t)
        metrics['nse'].append(nse_val)
        metrics['mse'].append(mse_val)
        metrics['rmse'].append(rmse_val)

        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        im1 = axes[0].imshow(targets_t, cmap='viridis', interpolation='nearest', vmin=global_min, vmax=global_max)
        axes[0].set_title(f"Ground Truth (t={t})")
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

        im2 = axes[1].imshow(preds_t, cmap='viridis', interpolation='nearest', vmin=global_min, vmax=global_max)
        axes[1].set_title(f"Prediction (t={t})")
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

        im3 = axes[2].imshow(residual_error, cmap='viridis', interpolation='nearest', vmin=0, vmax=residual_max)
        axes[2].set_title(f"Absolute Residual Error (t={t})")
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

        # Convert plot to image
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # Resize and append frame
        pil_frame = Image.fromarray(frame).resize(target_dpi_resolution, Image.LANCZOS)
        frames.append(np.array(pil_frame))
        plt.close(fig)

    # Save GIF if frames exist
    if frames:
        gif_path = os.path.join(report_path, gif_name)
        imageio.mimsave(gif_path, frames, duration=1)
        print(f"GIF saved successfully to {gif_path}")

    # Compute and return mean metrics
    return (
        np.nanmean(metrics['nse']),
        np.nanmean(metrics['mse']),
        np.nanmean(metrics['rmse'])
    )


def rmse_correlation_figs(
    valid_targets: torch.Tensor, 
    preds: torch.Tensor, 
    batch_idx: int, 
    step: str,  
    report_path: str, 
    no_value: float = -999
    
) -> None:
    """
    Generate spatial RMSE and correlation maps for model predictions.

    Args:
        valid_targets: Ground truth tensor
        preds: Predicted tensor
        batch_idx: Batch index for naming
        report_path: Directory to save results
        no_value: Value representing invalid data
    """
    def compute_cell_metrics(targets: np.ndarray, predictions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute per-cell RMSE and correlation.

        Args:
            targets: Ground truth array
            predictions: Predicted array

        Returns:
            Tuple of RMSE and correlation maps
        """
        cell_rmse_map = np.sqrt(np.mean((targets - predictions) ** 2, axis=0))
        
        cell_correlation_map = np.zeros_like(cell_rmse_map)
        for i in range(targets.shape[1]):
            for j in range(targets.shape[2]):
                cell_targets = targets[:, i, j]
                cell_preds = predictions[:, i, j]
                
                # Filter out NaN and invalid values
                valid_mask = ~(np.isnan(cell_targets) | np.isnan(cell_preds) | 
                               (cell_targets == no_value) | (cell_preds == no_value))
                
                if valid_mask.sum() > 1:
                    correlation = np.corrcoef(
                        cell_targets[valid_mask], 
                        cell_preds[valid_mask]
                    )[0, 1]
                    cell_correlation_map[i, j] = 0 if np.isnan(correlation) else correlation

        return cell_rmse_map, cell_correlation_map

    # Extract and process data for the first sample in the batch
    time_steps = preds.shape[1]
    all_preds = [preds[0, t, 0, :, :].cpu().detach().numpy() for t in range(time_steps)]
    all_targets = [valid_targets[0, t, 0, :, :].cpu().detach().numpy() for t in range(time_steps)]

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Compute metrics
    cell_rmse_map, cell_correlation_map = compute_cell_metrics(all_targets, all_preds)

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # RMSE Plot
    im1 = ax1.imshow(cell_rmse_map, cmap='viridis')
    ax1.set_title('RMSE per Cell', fontsize=14)
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, label='RMSE', fraction=0.046, pad=0.04)

    # Correlation Plot
    im2 = ax2.imshow(cell_correlation_map, cmap='coolwarm', vmin=-1, vmax=1)
    ax2.set_title('Correlation per Cell', fontsize=14)
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, label='Correlation', fraction=0.046, pad=0.04)

    # Save figure
    output_path = os.path.join(report_path, f"rmse_correlation_maps_batch_{batch_idx}_{step}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f'Batch {batch_idx + 1} - Step {step} - RMSE and Correlation maps saved to {output_path}')
