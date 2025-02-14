
import os 
import torch
import logging
import numpy as np
import time 
from typing import Dict, Any
from torch.utils.data import DataLoader, TensorDataset
import h5py


def clean_high_value_no_value_regions(target, dynamic=None, static=None, categorical=None, no_value=-999, new_no_value=-999):
    """
    Replace regions with no_value and high target values with new_no_value or zero.

    Args:
        target (np.ndarray): Target tensor
        dynamic (np.ndarray, optional): Dynamic tensor
        static (np.ndarray, optional): Static tensor
        categorical (np.ndarray, optional): Categorical tensor
        no_value (float, optional): Original no_value marker
        new_no_value (float, optional): New no_value marker

    Returns:
        tuple: Cleaned tensors
    """
    # Create a mask for high-value no_value regions
    high_value_mask = (target == no_value) & (target > 0.9)

    # Replace target values
    target[high_value_mask] = new_no_value

    # Process optional tensors
    if dynamic is not None:
        dynamic[high_value_mask] = new_no_value

    if static is not None:
        static[high_value_mask] = new_no_value

    if categorical is not None:
        categorical[high_value_mask] = 0

    return target, dynamic, static, categorical

def resample_to_250_to_30m(data_30m):
    # Dimensions of the input 30m resolution data
    input_shape = data_30m.shape
    #print("Original Shape (30m):", input_shape)

    # Resampling from 30m to 250m using nearest neighbor
    scale_factor_down = 250 // 30
    resampled_250m = data_30m[::scale_factor_down, ::scale_factor_down]
    #print("Downsampled Shape (250m):", resampled_250m.shape)

    # Resampling back from 250m to 30m using nearest neighbor
    scale_factor_up = 30 // 250
    resampled_back_30m = np.kron(resampled_250m, np.ones((scale_factor_down, scale_factor_down)))

    # Crop or pad to ensure the final size matches the original input size
    final_resampled_30m = resampled_back_30m[:input_shape[0], :input_shape[1]]
    #print("Final Shape (Resampled Back to 30m):", final_resampled_30m.shape)

    assert final_resampled_30m.shape == input_shape, "Final shape does not match the original input shape"

    return final_resampled_30m
def plot_single_distribution(data, name, no_value):
    import matplotlib.pyplot as plt
    #data = np.where(data == 0, np.nan, data)
    data = np.where(data == no_value, np.nan, data)
    plt.hist(data.flatten(), bins=100)
    plt.title(f"Distribution of {name}")
    os.makedirs("figs", exist_ok=True)
    plt.savefig(f"figs/{name}.png", dpi=300)
    plt.close()


def get_swatplus_names(config):
    swatplus_output_path = config['swatplus_output_path']
    print(f"Reading SWAT+ output datasets from {swatplus_output_path}")
    with h5py.File(swatplus_output_path, "r") as f:
        NAMES = list(f.keys())
        ### if less than 200 batches in each NAME, remove the NAME
        #NAMES = [name for name in NAMES if f[f'{name}/dynamic'].shape[0] > 200]
        print(f"Available {len(NAMES)} SWAT+ output datasets.")
    return NAMES
def replace_no_value(data, no_value, new_no_value):
    """
    Replace 'no_value' in the data with a new specified value.
    """
    data[data == no_value] = new_no_value
    return data

def swat_model_size(path, name):
    """
    Helper function to check the size of a dataset in an HDF5 file.
    """
    #print("######## check_size called ########")
    with h5py.File(path, "r") as f:
        data = f[f"{name}/train/feature_tensors"]   
        #print(f"size of {name} is {data.shape}")
        return data.shape

def create_dataloader(dynamic, static, categorical, targets, batch_size, shuffle=True):
    """
    Create a PyTorch DataLoader from input and target tensors.
    """
    #inputs = torch.tensor(inputs).clone().detach().float()
    #targets = torch.tensor(targets).clone().detach().float()
    dynamic = torch.tensor(dynamic).clone().detach().float()
    static = torch.tensor(static).clone().detach().float()
    categorical = torch.tensor(categorical).clone().detach().float()
    targets = torch.tensor(targets).clone().detach().float()
    
    #print("dynamic shape:", dynamic.shape)
    #print("static shape:", static.shape)
    #print("categorical shape:", categorical.shape)
    #print("targets shape:", targets.shape)

    dataset = TensorDataset(dynamic, static, categorical, targets)
    
    #dataset = TensorDataset(inputs, targets)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=False,
        prefetch_factor=6,
    )


def chunk_sequences(data_dict, seq_len):
    """
    Chunk the data into sequences of length seq_len.
    Ensures consistency across all tensors even when there's only one time step.
    
    Args:
        data_dict (dict): Dictionary containing numpy arrays to chunk, e.g.,
                          {"dynamic": dynamic_array, "static": static_array, 
                           "target": target_array, "categorical": categorical_array}.
        seq_len (int): Sequence length to chunk the data into.
        
    Returns:
        dict: Dictionary with chunked arrays.
    """
    dynamic_array = data_dict["dynamic"][:]
    num_samples, time_steps, *dims = dynamic_array.shape
    num_chunks = (time_steps + seq_len - 1) // seq_len  # Ensure enough chunks are generated

    # Initialize chunked arrays
    chunked_data = {key: [] for key in data_dict.keys()}

    for i in range(num_chunks):
        start_idx = i * seq_len
        end_idx = min((i + 1) * seq_len, time_steps)

        for key, array in data_dict.items():
            if len(array.shape) > 2:  # For arrays with time dimension
                chunk = array[:, start_idx:end_idx, ...]

                # If the chunk is shorter than seq_len, pad it
                if chunk.shape[1] < seq_len:
                    pad_shape = (chunk.shape[0], seq_len - chunk.shape[1], *chunk.shape[2:])
                    chunk = np.concatenate([chunk, np.zeros(pad_shape, dtype=array.dtype)], axis=1)

                chunked_data[key].append(chunk)
            else:  # For arrays without time dimension (e.g., static)
                chunked_data[key].append(array)

    # Concatenate along the batch dimension
    for key in chunked_data:
        chunked_data[key] = np.concatenate(chunked_data[key], axis=0)

    return chunked_data


def plot_loss_over_epochs(losses, val_losses, report_path, total_norms):
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot training and validation losses on the first y-axis
    ax1.plot(range(2, len(losses)), losses[2:], color='blue', label='Training Loss')
    ax1.plot(range(2, len(val_losses)), val_losses[2:], color='red', label='Validation Loss')
    ax1.set_xlabel('Epoch')
    set_axis_labels(ax1, 'Loss', 'blue')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # Create a second y-axis for the gradient norms
    ax2 = ax1.twinx()
    ax2.plot(range(2, len(total_norms)), total_norms[2:], color='green', label='Gradient Norm')
    ax2.set_ylabel('Gradient Norm', color='green')
    set_axis_labels(ax2, 'Gradient Norm', 'green')

    # Add a title
    plt.title('Loss and Gradient Norms over Epochs')
    plt.tight_layout()  
    # Save the plot
    plt.savefig(f"{report_path}/loss_and_norms_over_epoch.png", dpi=300)
    plt.close()

def set_axis_labels(axis, label, color):
    axis.set_ylabel(label, color=color)
    axis.tick_params(axis='y', labelcolor=color)


class EarlyStopping:
    """
    class EarlyStopping:
    - Implements early stopping with a warm-up phase.
    - Restores the best model weights.
    - Checks for overfitting by comparing recent trends in train and validation losses.
    """
    def __init__(self, config, logger=None):
        self.patience = config.get('early_stopping_patience', 10)
        self.min_delta = config.get('early_stopping_min_delta', 1e-6)
        self.num_warmup_steps = config.get('num_warmup_steps', 10)
        self.restore_best_weights = config.get('restore_best_weights', True)
        self.overfit_threshold = config.get('overfit_threshold', 1e-3)
        self.verbose = True
        self.logger = logger

        # Window size for checking overfitting
        self.overfitting_window = config.get('overfitting_window', 3)
        self.check_overfitting_after = config.get('check_overfitting_after', 10)

        self.val_losses = []
        self.train_losses = []

        # Internal state
        self.best_loss = float('inf')
        self.best_epoch = -1
        self.counter = 0
        self.best_weights = None  # To store the best model weights

    def _is_overfitting(self):
        """
        Check if the model is overfitting by comparing the recent average validation loss
        with the recent average training loss. If validation loss is consistently higher 
        than training loss, we consider it overfitting.
        """
        if len(self.val_losses) <= self.check_overfitting_after:
            return False  # Not enough data to decide

        # Extract recent losses
        recent_val = self.val_losses[-self.overfitting_window:]
        recent_train = self.train_losses[-self.overfitting_window:]

        # Calculate averages
        avg_val = np.mean(recent_val)
        avg_train = np.mean(recent_train)

        # Consider overfitting if validation is consistently worse than train
        # You can adjust the criteria, e.g., avg_val > avg_train + some_threshold
        return avg_val > avg_train + self.overfit_threshold

    def check_early_stopping(self, val_loss, train_loss, epoch, model=None):
        """
        Check whether early stopping should be triggered based on validation loss 
        and overfitting criteria.
        """
        # Warm-up phase
        if epoch < self.num_warmup_steps:
            if self.verbose:
                self.logger.info(f"Epoch {epoch}: Warm-up phase, early stopping not active.", time_stamp=False)
            self.val_losses.append(val_loss)
            self.train_losses.append(train_loss)
            return False, False  # Continue training

        # Record current losses
        self.val_losses.append(val_loss)
        self.train_losses.append(train_loss)

        # Check for overfitting
        if self._is_overfitting():
            if self.verbose:
                self.logger.info(
                    "Early stopping triggered due to overfitting.",
                    time_stamp=False,
                )
            return True, False

        # Check for improvement
        if val_loss < self.best_loss - self.min_delta:
            return self._extracted_from_check_early_stopping_30(val_loss, epoch, model)
        # No significant improvement
        self.counter += 1
        if self.verbose:
            self.logger.info(f"Epoch {epoch}: Validation loss did not improve. Patience counter {self.counter}/{self.patience}.", time_stamp=False)
        # Check patience
        if self.counter >= self.patience:
            if self.verbose:
                self.logger.info(f"Early stopping triggered at epoch {epoch}. Best loss: {self.best_loss:.6f} at epoch {self.best_epoch}.", time_stamp=False)
            if model is not None and self.restore_best_weights:
                if self.verbose:
                    self.logger.info("Restoring best model weights.", time_stamp=False)
                model.load_state_dict(self.best_weights)
            return True, False

        return False, False

    # TODO Rename this here and in `check_early_stopping`
    def _extracted_from_check_early_stopping_30(self, val_loss, epoch, model):
        # Improved validation loss
        self.best_loss = val_loss
        self.best_epoch = epoch
        self.counter = 0
        if model is not None and self.restore_best_weights:
            self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        if self.verbose:
            self.logger.info(f"Epoch {epoch}: Validation loss improved to {val_loss:.6f}. Resetting patience counter.", time_stamp=False)
        return False, True  # Continue training with a new best model


def calculate_gradient_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    return total_norm



def generate_model_name(config: Dict[str, Any]) -> str:
    """
    Generate a unique model name based on configuration parameters.
    
    Args:
        config (dict): Configuration dictionary containing model parameters.
    
    Returns:
        str: Generated model name.
    """
    # Common parameters
    base_name = f"{config['target_array']}_{config['model']}"
    common_params = [
        f"bs{config['batch_size']}",
        f"bw{config.get('batch_window', 'default')}",
        f"lr{config['opt_lr']}",
        f"wd{config['weight_decay']}"
    ]

    # Model-specific parameters
    if "Transformer" in config['model']:
        transformer_params = [
            f"epochs{config.get('num_training_epochs', 'default')}",
            f"nh{config.get('num_heads', 'default')}",
            f"nl{config.get('num_layers', 'default')}",
            f"fe{config.get('forward_expansion', 'default')}",
            f"embs{config.get('embed_dim', 'default')}",
            f"dropout{config.get('dropout', 'default')}"
        ]
        params = transformer_params
    else:
        other_params = [
            f"sl{config.get('seq_len', 'default')}",
            f"{config.get('scheduler', 'default')}",
            f"{config.get('loss_function', 'default')}"
        ]
        params = other_params

    # Combine all parameters
    full_name = "_".join([base_name] + common_params + params)
    
    # Truncate if too long and ensure filename safety
    max_length = 255
    safe_name = "".join(c if c.isalnum() or c in ['_', '-'] else '_' for c in full_name)
    return safe_name[:max_length]


def prepare_classification(config, preds_chunk, targets_chunk, no_value) -> tuple:
    """
    Prepares predictions and targets for classification by masking and reshaping.
    """
    preds_chunk = preds_chunk.permute(0, 2, 3, 4, 1).reshape(-1, config['num_classes'])
    targets_chunk_flat = targets_chunk.reshape(-1)
    mask = (targets_chunk_flat != no_value)
    preds_chunk = preds_chunk[mask]
    targets_chunk_flat = targets_chunk_flat[mask]
    targets_chunk_flat = targets_chunk_flat.long()
    ### remove no values
    valid_mask = (targets_chunk_flat >= 0) & (targets_chunk_flat < config['num_classes'])
    preds_chunk = preds_chunk[valid_mask]
    targets_chunk_flat = targets_chunk_flat[valid_mask]
    return preds_chunk, targets_chunk_flat

class LoggerSetup:
    def __init__(self, report_path, verbose=True, rewrite=False):
        """
        Initialize the LoggerSetup class.

        Args:
            report_path (str): Path to the directory where the log file will be saved.
            verbose (bool): Whether to print logs to console. Defaults to True.
        """
        self.report_path = report_path
        self.logger = None
        self.verbose = verbose
        self.rewrite = rewrite  

    def setup_logger(self, name="GeoClassCNNLogger"):
        """
        Set up the logger to log messages to a file and optionally to the console.

        Returns:
            logging.Logger: Configured logger.
        """
        if not self.logger:
            # Define the path for the log file
            path = os.path.join(self.report_path, f"{name}.log")
            if self.rewrite and os.path.exists(path):
                os.remove(path)
            # Create a logger
            self.logger = logging.getLogger(name)
            self.logger.setLevel(logging.INFO)  # Set the logging level

            # FileHandler for logging to a file
            file_handler = logging.FileHandler(path)
            file_handler.setLevel(logging.INFO)
            self.logger.addHandler(file_handler)

            # Conditionally add console handler based on verbose flag
            if self.verbose:
                console_handler = logging.StreamHandler()
                console_handler.setLevel(logging.INFO)
                self.logger.addHandler(console_handler)

            self.logger.info(f"Logging to {path}")

        return self.logger
    def error(self, message, time_stamp=True):
        """
        Log an error message.

        Args:
            message (str): The error message to log.
            time_stamp (bool): Whether to include a timestamp in the log.
        """
        self.info(message, level="error", time_stamp=time_stamp)

    def warning(self, message, time_stamp=True):
        """
        Log a warning message.

        Args:
            message (str): The warning message to log.
            time_stamp (bool): Whether to include a timestamp in the log.
        """
        self.info(message, level="warning", time_stamp=time_stamp)

    def info(self, message, level="info", time_stamp=True):
        """
        Log a message with or without a timestamp.

        Args:
            message (str): The message to log.
            level (str): The logging level (e.g., "info", "error").
            time_stamp (bool): Whether to include a timestamp in the log.
        """
        # Create a temporary logger with the desired format
        temp_logger = logging.getLogger("TempLogger")
        temp_logger.setLevel(self.logger.level)

        # Remove existing handlers to avoid duplicates
        temp_logger.handlers.clear()

        # Define the log format based on the time_stamp flag
        log_format = '%(asctime)s - %(levelname)s - %(message)s' if time_stamp else '%(levelname)s - %(message)s'

        # Add file handler
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                new_file_handler = logging.FileHandler(handler.baseFilename)
                new_file_handler.setFormatter(logging.Formatter(log_format))
                temp_logger.addHandler(new_file_handler)

        # Conditionally add console handler based on verbose flag
        if self.verbose:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(log_format))
            temp_logger.addHandler(console_handler)

        # Log the message at the specified level
        log_methods = {
            "info": temp_logger.info,
            "error": temp_logger.error,
            "warning": temp_logger.warning,
            "debug": temp_logger.debug
        }
        log_method = log_methods.get(level.lower(), temp_logger.info)
        log_method(message)

# Usage example
if __name__ == "__main__":
    report_path = "ml_data"
    os.makedirs(report_path, exist_ok=True)  # Ensure the directory exists

    # Create a LoggerSetup instance
    logger = LoggerSetup(report_path)
    logger.setup_logger()

    # Log with timestamp
    logger.info("Test logging with timestamp")

    # Log without timestamp
    logger.info("Test logging without timestamp", time_stamp=False)
