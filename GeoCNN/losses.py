import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit

def weighted_cross_entropy_loss(num_classes, zero_weight=0.01, device="cpu"):
    """
    Returns a CrossEntropyLoss with a specified weight for class 0.

    Args:
        num_classes (int): Number of classes in the classification task.
        zero_weight (float): Weight for class 0. All other classes get a weight of 1.
        device (str): The device where the weights tensor should be located.
        
    Returns:
        CrossEntropyLoss: Weighted CrossEntropyLoss instance.
    """
    # Define weights: 0.01 for class 0, 1 for all other classes
    weights = torch.ones(num_classes, device=device)
    weights[0] = zero_weight  # Class 0 gets lower weight
    return nn.CrossEntropyLoss(weight=weights)


def HuberLossWithThreshold(y_pred, y_true, delta=1.0, threshold=1e-3):
    """
    Huber Loss with small-value thresholding for regression.

    Args:
        y_pred (torch.Tensor): Predicted values.
        y_true (torch.Tensor): Ground truth values.
        delta (float): Huber loss threshold.
        threshold (float): Minimum value threshold for small-value elimination.
    
    Returns:
        torch.Tensor: Computed Huber loss.
    """
    # Mask out small target values
    mask = y_true >= threshold
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    # Compute Huber loss
    error = y_true - y_pred
    is_small_error = torch.abs(error) <= delta

    # Apply piecewise loss
    loss = torch.where(
        is_small_error,
        0.5 * error ** 2,  # Squared loss
        delta * (torch.abs(error) - 0.5 * delta)  # Linear loss
    )

    return torch.mean(loss)



def LogLoss(y_pred, y_true, epsilon=1e-7, threshold=1e-3):
    """
    Custom Log Loss function with small-value elimination.

    Args:
        y_pred (torch.Tensor): Predicted values (output of the model, probabilities).
        y_true (torch.Tensor): Ground truth values (target labels).
        epsilon (float): Small constant to avoid log(0) issues.
        threshold (float): Minimum value threshold to filter out small values.
    
    Returns:
        torch.Tensor: Computed log loss.
    """

    # Avoid log(0) by clamping predictions and targets to a small value
    y_pred = torch.clamp(y_pred, min=epsilon, max=1.0 - epsilon)
    y_true = torch.clamp(y_true, min=epsilon, max=1.0)

    # Mask out small values in `y_true` (optional for `y_pred` if needed)
    mask = y_true >= threshold
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    # Compute the binary log loss
    loss = -torch.mean(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))

    print(f"LogLoss: {loss.item()}")
    return loss

class WeightedLogCoshLoss(nn.Module):
    def __init__(self, zero_weight=1.0, non_zero_weight=5.0):
        super(WeightedLogCoshLoss, self).__init__()
        self.zero_weight = zero_weight
        self.non_zero_weight = non_zero_weight

    def forward(self, predictions, targets):
        # Identify zero and non-zero targets
        is_zero = (targets == 0).float()
        is_non_zero = 1.0 - is_zero

        # Log-Cosh Loss (smooth for large values)
        log_cosh_loss = torch.log(torch.cosh(predictions - targets))
        
        # Weighted loss
        weighted_loss = (
            self.zero_weight * is_zero * log_cosh_loss +
            self.non_zero_weight * is_non_zero * log_cosh_loss
        )

        # Return mean loss
        return weighted_loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, gamma=2., alpha=0.25):
        """
        Focal Loss as defined in:
        "Focal Loss for Dense Object Detection" (https://arxiv.org/abs/1708.02002)
        
        Args:
            gamma (float): Focusing parameter. Default is 2.
            alpha (float): Balance parameter. Default is 0.25.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        """
        Compute the focal loss.
        
        Args:
            y_pred (torch.Tensor): Predicted probabilities (logits after sigmoid). Shape [batch_size, ...].
            y_true (torch.Tensor): Ground truth labels. Shape [batch_size, ...].
        
        Returns:
            torch.Tensor: Computed focal loss.
        """
        epsilon = 1e-8  # To prevent division by zero
        y_pred = torch.clamp(y_pred, epsilon, 1.0 - epsilon)
        y_true = y_true.float()

        # Alpha adjustment
        alpha_t = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)

        # Probability for the target class
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)

        # Focal loss computation
        loss = -alpha_t * torch.pow(1 - p_t, self.gamma) * torch.log(p_t)
        return loss.mean()

class SpatioTemporalLoss(nn.Module):
    def __init__(self, config, alpha=0.007, beta=0.016, c=1, omega_o=0.57, omega_t=0.41):
        """
        Quantile Weighted Mean Squared Error Loss with Boundary Emphasis and Seasonal Temporal Weighting.
        """
        super(SpatioTemporalLoss, self).__init__()
        self.config = config  
        self.alpha = alpha
        self.beta = beta
        self.c = c
        self.omega_o = omega_o
        self.omega_t = omega_t
        self.no_value = config['new_no_value']

    def compute_boundary_mask(self, tensor):
        """
        Generate a weighted boundary mask for the input tensor with decreasing weights.

        Args:
            tensor (torch.Tensor): Input tensor of shape [batch, time, channels, height, width].

        Returns:
            torch.Tensor: Weighted boundary mask with the same shape as the input tensor.
        """
        boundary_mask = torch.zeros_like(tensor)
        weights = [1.0, 0.98, 0.97, 0.96, 0.95]  # Weights for each row/column layer

        # Top and Bottom boundaries (with weights)
        for i, weight in enumerate(weights):
            boundary_mask[ :, :, i, :] += weight  # Top boundary
            boundary_mask[:, :, -i-1, :] += weight  # Bottom boundary

        # Left and Right boundaries (with weights)
        for i, weight in enumerate(weights):
            boundary_mask[ :, :, :, i] += weight  # Left boundary
            boundary_mask[ :, :, :, -i-1] += weight  # Right boundary

        # Handle corners: explicitly sum contributions and divide by 2 to normalize
        # Handle corners: explicitly sum contributions and apply a corner weight
        corner_weight = 1  # Assign higher importance to corners
        for i, weight in enumerate(weights):
            boundary_mask[:, :, i, i] += corner_weight * weight  # Top-left corner
            boundary_mask[:, :, i, -i-1] += corner_weight * weight  # Top-right corner
            boundary_mask[:, :, -i-1, i] += corner_weight * weight  # Bottom-left corner
            boundary_mask[:, :, -i-1, -i-1] += corner_weight * weight  # Bottom-right corner

        return boundary_mask

    def _classify_categories(self, y_true):
        """
        Classify data into categories: low, outlier, normal.
        """
        no_value_mask = y_true == self.no_value
        #print(f"number of no values: {torch.sum(low_mask)}")    
        ### based on 95th percentile
        
        outlier_mask = y_true > torch.quantile(y_true, 0.90)
        #outlier_mask = (y_true >0.95)

        normal_mask = ~(no_value_mask | outlier_mask)

        return no_value_mask, outlier_mask, normal_mask


    def compute_torrential_loss(self, y_pred_t, y_true_t, normal_mask):
    
        weights = self.alpha * torch.exp(self.beta * y_true_t**self.c)
        # Masks
        torrential_mask = y_true_t >= torch.quantile(y_true_t[y_true_t!=self.no_value], 1/3)
        #torrential_mask = y_true_t >= 0.05
        over_mask = y_pred_t >= y_true_t
        under_mask = y_pred_t < y_true_t

        ### apply normal mask
        over_mask = over_mask & normal_mask
        under_mask = under_mask & normal_mask
        torrential_mask = torrential_mask & normal_mask

        # Overall loss
        loss_overall = (
            torch.sum((1 - self.omega_o) * torch.abs(y_true_t[over_mask] - y_pred_t[over_mask])) +
            torch.sum(self.omega_o * torch.abs(y_true_t[under_mask] - y_pred_t[under_mask]))
        ) / y_true_t.numel()

        # Torrential rain loss
        loss_torrential = (
            torch.sum((1 - self.omega_t) * weights[torrential_mask & over_mask] *
                    (y_true_t[torrential_mask & over_mask] - y_pred_t[torrential_mask & over_mask])**2) +
            torch.sum(self.omega_t * weights[torrential_mask & under_mask] *
                    (y_true_t[torrential_mask & under_mask] - y_pred_t[torrential_mask & under_mask])**2)
        ) / y_true_t.numel()

        return loss_overall , loss_torrential

    def compute_loss_per_timestep(self, y_pred_t, y_true_t):
        """
        Compute the loss for a single time step.
        """
        low_mask, outlier_mask, normal_mask = self._classify_categories(y_true_t)
        # Low loss
        low_loss = torch.sum((self.omega_o)*torch.abs(y_true_t[low_mask] - y_pred_t[low_mask])) / y_true_t.numel()
        
        # Outlier loss
        outlier_loss = torch.sum((1-self.omega_o)*torch.abs(y_true_t[outlier_mask] - y_pred_t[outlier_mask])) / y_true_t.numel()  

        # Boundary loss
        boundary_mask = self.compute_boundary_mask(y_true_t)

        boundary_loss = (
            torch.sum(boundary_mask * ((1 - self.omega_o) * torch.abs(y_true_t - y_pred_t))) / boundary_mask.numel()
        ) 

        # Compute torrential loss
        torrential_loss, loss_overall = self.compute_torrential_loss(y_pred_t, y_true_t, normal_mask)
        #print(f"low_loss: {low_loss.item()}, outlier_loss: {outlier_loss.item()}, boundary_loss: {boundary_loss.item()}, torrential_loss: {torrential_loss.item()}, loss_overall: {loss_overall.item()}")
        
        return low_loss + outlier_loss + boundary_loss + torrential_loss + loss_overall

    def get_season_indices(self):
        """
        Assigns indices of timesteps to corresponding seasons based on their position in the sequence.
        
        Returns:
            dict: Mapping of season names to indices corresponding to time steps.
        """
        time_steps = self.config['seq_len']  # Total number of timesteps in the sequence

        # Define seasonal ranges
        months_to_seasons = {
            "Winter": [11, 0, 1],  # Dec, Jan, Feb
            "Spring": [2, 3, 4],   # Mar, Apr, May
            "Summer": [5, 6, 7],   # Jun, Jul, Aug
            "Fall": [8, 9, 10]     # Sep, Oct, Nov
        }

        # Generate the seasonal indices
        season_indices = {season: [] for season in months_to_seasons}

        for t in range(time_steps):
            month = t % 12  # Cycle through months (assuming monthly data, 12 months per year)
            for season, months in months_to_seasons.items():
                if month in months:
                    season_indices[season].append(t)
                    break

        return season_indices

                    
    def calculate_seasonal_error(self, y_pred, y_true, season_indices):
        """
        Calculate errors for each season.
        
        Args:
            y_pred (torch.Tensor): Predicted values of shape [batch, time, channels, height, width].
            y_true (torch.Tensor): Ground truth values of shape [batch, time, channels, height, width].
            season_indices (dict): Mapping of season names to indices corresponding to time steps.
            
        Returns:
            dict: Seasonal errors (e.g., mean absolute error) for each season.
        """
        seasonal_errors = {}
        for season, indices in season_indices.items():
            # Select time steps corresponding to the season
            y_pred_season = y_pred[:, indices]
            y_true_season = y_true[:, indices]

            ### flatten and remove no values
            y_pred_season = y_pred_season.flatten()
            y_true_season = y_true_season.flatten()
            mask = y_true_season != self.no_value
            y_pred_season = y_pred_season[mask]
            y_true_season = y_true_season[mask]
            
            # Compute error for the season (e.g., Mean Absolute Error)
            seasonal_error = torch.mean(torch.abs(y_pred_season - y_true_season))
            seasonal_errors[season] = seasonal_error.item()
        
        return seasonal_errors


    def forward(self, y_pred, y_true):
        """
        Compute the combined loss for all time steps and calculate seasonal errors.
        
        Args:
            y_pred (torch.Tensor): Predicted values of shape [batch, time, channels, height, width].
            y_true (torch.Tensor): Ground truth values of shape [batch, time, channels, height, width].
        """
        time_steps = y_pred.size(1)

        # Define seasonal indices (e.g., time steps for Winter, Spring, Summer, Autumn)
        season_indices = dict(self.get_season_indices())

        #print(f"winter indices: {season_indices['Winter']}")
        # Use torch.jit.fork to compute loss for each timestep in parallel
        futures = [
            torch.jit.fork(self.compute_loss_per_timestep, y_pred[:, t], y_true[:, t])
            for t in range(time_steps)
        ]

        # Wait for all computations to complete and gather results
        losses = torch.stack([torch.jit.wait(future) for future in futures])

        # Compute seasonal errors
        seasonal_errors = self.calculate_seasonal_error(y_pred, y_true, season_indices)

        return torch.mean(losses) + sum(seasonal_errors.values())
