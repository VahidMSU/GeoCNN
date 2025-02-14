import os
import time
import torch
from typing import Dict
from GeoCNN.registry import (
    select_model,
      select_criterion, 
     select_scheduler,
       setup_optimizer)

from GeoCNN.utils import (
    LoggerSetup,
    EarlyStopping, 
    generate_model_name, 
    plot_loss_over_epochs
)
from GeoCNN.prediction import run_prediction_evaluation
import torch
from GeoCNN.pipeline import DataPipeline

class GeoClassCNN:
    def __init__(self, config):
        # Initialize the model
        self.config = config
        self.initialization()
        self.import_data()
        self.training_setup()
        

    def initialization(self):
        # Explicitly assign only the necessary config keys to attributes
        required_keys = ['hydrogeodataset_path', 'output_path', 'gpu_index', 'seq_len', 'accumulation_steps', 'batch_size', 'num_training_epochs', 'max_norm']
        for key in required_keys:
            setattr(self, key, self.config[key])

        # Validate and set up paths
        assert os.path.exists(self.hydrogeodataset_path), f"Database path {self.hydrogeodataset_path} does not exist"
        self.model_name = self.config.get('model_name', generate_model_name(self.config))
        self.report_path = os.path.join(self.output_path, self.model_name)
        self.config['report_path'] = self.report_path
        self.best_model_path = os.path.join(self.report_path, "best_model.pth")
        os.makedirs(self.report_path, exist_ok=True)

        # Set up logging
        self.logger = LoggerSetup(self.report_path)
        self.logger.setup_logger("GeoCNNTrainer")

        # Set up GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_index)
        self.device = torch.device('cuda')
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available.")
        self.logger.info(f"Using GPU: {self.gpu_index}")

        # Initialize parameters
        self.best_loss = float('inf')
        self.model = None
        self.gradient_norms = []
        self.losses = []
        self.val_losses = []
        self.total_norms = []
        self.learning_rates = []    
        self.num_total_optimization_steps = []

        # Save configuration for reproducibility
        config_path = os.path.join(self.report_path, "config.txt")
        with open(config_path, "w") as f:
            for key, value in self.config.items():
                f.write(f"{key}: {value}\n")


    def import_data(self):
        self.data_pipeline = DataPipeline(self.config)
        self.logger.info("Data pipeline initialized")   
        self.data_pipeline.update_loaders(step=0 if self.config["preloading"] else None)
        self.logger.info("Data pipeline loaders updated")
        self.num_channels = self.data_pipeline.active_data["num_channels"]
        self.num_dynamic_channels = self.data_pipeline.active_data["num_dynamic_channels"]
        self.num_static_channels = self.data_pipeline.active_data["num_static_channels"]
        self.num_categorical_channels = self.data_pipeline.active_data["num_categorical_channels"]
        self.config["num_channels"] = self.num_channels
        self.config["num_dynamic_channels"] = self.num_dynamic_channels
        self.config["num_static_channels"] = self.num_static_channels
        self.config["num_categorical_channels"] = self.num_categorical_channels

        self.effective_opt_steps_per_load = self.data_pipeline.active_data["steps_per_load"]//self.accumulation_steps
        self.is_single_dataset = len(self.data_pipeline.deload_steps) == 1  # Add this flag
        print("##################### Single Dataset Flag: ", self.is_single_dataset)

        self.total_effective_steps = self.effective_opt_steps_per_load * self.num_training_epochs * len(self.data_pipeline.deload_steps)

    def training_setup(self):
        """
        setup model, optimizer, scheduler, criterion, early stopping and scaler
        """
        self.model = select_model(self.config, device=self.device)
        self.optimizer = setup_optimizer(self.config, self.model)
        self.scheduler = select_scheduler(self.optimizer, self.config, self.total_effective_steps)
        self.criterion = select_criterion(self.config, self.device) # Loss function
        self.early_stopping = EarlyStopping(self.config, self.logger)   
        self.scaler = torch.amp.GradScaler()
        

    def predict(self) -> None:
        try:
            # Ensure test_loader is created before loading model
            self.logger.info(f"Loading model from {self.best_model_path}")  
            self.model = torch.load(self.best_model_path, map_location=self.device)
            # If the model was wrapped in DataParallel during training
            if isinstance(self.model, torch.nn.DataParallel):
                self.model = self.model.module
            self.model.to(self.device)
            self.model.eval()
            for deload_step in range(len(self.data_pipeline.deload_steps)):
                self.logger.info(f"Processing {deload_step + 1}/{len(self.data_pipeline.deload_steps)}")
                self.test_loader = self.data_pipeline.get_active_test_loaders()
                self.no_value = 1e-6
                run_prediction_evaluation(self.config, self.model, self.test_loader, self.report_path, self.no_value, self.device, deload_step)
                
                if not self.is_single_dataset:  # Only deload if multiple datasets
                    self.data_pipeline.deload_data(step=deload_step, stage="test")
        except FileNotFoundError:
            self.logger.info("No model found for prediction")
        finally:
            self.data_pipeline.stop_preloading()

    def plot_learning_rate_optimization_steps(self):
        """
        Plot learning rate and gradient norms over cumulative optimization steps.
        """
        import matplotlib.pyplot as plt

        # Ensure num_total_optimization_steps is cumulative
        cumulative_steps = torch.cumsum(torch.tensor(self.num_total_optimization_steps), dim=0)

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        # Plot learning rate and gradient norms
        ax1.plot(cumulative_steps, self.learning_rates, 'g-', label='Learning Rate')
        ax2.plot(cumulative_steps, self.gradient_norms, 'b-', label='Gradient Norm')

        ax1.set_xlabel('Cumulative Optimization Steps')
        ax1.set_ylabel('Learning Rate', color='g')
        ax2.set_ylabel('Gradient Norm', color='b')

        plt.title("Learning Rate and Gradient Norm over Cumulative Optimization Steps")
        plt.grid(linestyle='--', linewidth=0.5, color='gray')
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(self.report_path, "learning_rate_gradient_norm.png"), dpi=300)
        plt.close()


    def optimization_step(self):

        # Unscale the gradients before clipping 
        self.scaler.unscale_(self.optimizer)
        total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_norm)
        
        if total_norm < 1e-6:
            self.logger.info("############## Gradient is zero ###############")
            return  # Skip optimizer step
        
        self.gradient_norms.append(total_norm.item())
        # Perform optimizer step and clear gradients
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.config["scheduler"] != "ReduceLROnPlateau":
            self.scheduler.step()
        else:
            if self.val_losses:
                self.scheduler.step(metrics=self.val_losses[-1])
        self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
        #self.logger.info(f"Optimizer step: {self.optimizer_steps}, LR: {self.optimizer.param_groups[0]['lr']:.2e}, Grad Norm: {total_norm:.4f}", time_stamp=False)
        self.optimizer.zero_grad(set_to_none=True)  # Zero gradients only after an optimizer step
        self.optimizer_steps += 1
        self.num_total_optimization_steps.append(self.optimizer_steps)

    def model_train_forward(self, deload_step, total_samples, epoch_loss):
        """ 
        Forward pass for the model with gradient accumulation.
        """
        train_loader = self.data_pipeline.get_active_train_loaders()

        for batch_idx, (dynamic_inputs, static_inputs, categorical_inputs, targets) in enumerate(train_loader):
            batch_loss = 0.0  # Accumulate loss for the batch
            #total_samples += inputs.size(0)  # Update total samples
            total_samples += dynamic_inputs.size(0)  # Update total samples

            ## skip if less than batch size
            if dynamic_inputs.size(0) < self.batch_size:
                self.logger.warning(f"Train Batch {batch_idx + 1} has less than batch size, skipping")
                continue

            # Move inputs and targets to device once per batch
            #inputs, targets = inputs.to(self.device), targets.to(self.device)
            dynamic_inputs, static_inputs, categorical_inputs, targets = dynamic_inputs.to(self.device), static_inputs.to(self.device), categorical_inputs.to(self.device), targets.to(self.device)

            with torch.amp.autocast(device_type='cuda'):
                preds = self.model(dynamic_inputs, static_inputs, categorical_inputs)
                loss = self.criterion(preds, targets)
                print(f"Stage: train, Step {deload_step} Batch {batch_idx + 1} Loss: {loss:.6f}")

            # Scale the loss for gradient accumulation
            scaled_loss = loss / self.accumulation_steps
            self.scaler.scale(scaled_loss).backward()
            batch_loss += loss.item()

            # Perform optimizer step after accumulation_steps batches or at the end of epoch
            if (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                self.optimization_step()

            # Accumulate raw batch loss for the epoch
            epoch_loss += batch_loss

            # Free unused GPU memory
            #self.cleanup(inputs, targets, preds, loss)

        if not self.is_single_dataset:  # Only deload if multiple datasets
            self.data_pipeline.deload_data(step=deload_step, stage="train")

        return epoch_loss, total_samples

    def train_loop(self) -> float:
        """
        Training loop for a single epoch with gradient accumulation and correct loss averaging.
        """
        self.model.train()
        epoch_loss = 0.0  # Total loss for the epoch
        total_samples = 0  # Total number of samples processed in the epoch

        # Reset gradients initially
        self.optimizer.zero_grad(set_to_none=True)
        self.optimizer_steps = 0

        for deload_step in range(len(self.data_pipeline.deload_steps)):
            self.logger.info(f"Processing {deload_step + 1}/{len(self.data_pipeline.deload_steps)}")
        
            epoch_loss, total_samples = self.model_train_forward(deload_step, total_samples, epoch_loss)
            self.logger.info(f"Train {deload_step} Loss: {epoch_loss:.6f} total_samples: {total_samples}")
            
        # Normalize total loss by the number of samples
        epoch_loss /= total_samples

        # Log diagnostics
        self.total_norm = torch.mean(torch.tensor(self.gradient_norms)).item()
        self.total_norms.append(self.total_norm)
        self.losses.append(epoch_loss)
        self.logger.info("End of Training for epoch")
        
        return epoch_loss

    def model_val_forward(self,deload_step, total_samples, val_loss): 
        val_loader = self.data_pipeline.get_active_val_loaders()             
        for batch_idx, (dynamic_inputs, static_inputs, categorical_inputs, targets) in enumerate(val_loader):   
            total_samples += dynamic_inputs.size(0)  # Update total samples
            if dynamic_inputs.size(0) < (self.batch_size//2)+2:
                self.logger.warning(f"Val Batch {batch_idx + 1} has less than batch size, skipping")
                continue
            # Move inputs and targets to device once per batch
            #inputs, targets = inputs.to(self.device), targets.to(self.device)
            dynamic_inputs, static_inputs, categorical_inputs, targets = dynamic_inputs.to(self.device), static_inputs.to(self.device), categorical_inputs.to(self.device), targets.to(self.device)


            with torch.amp.autocast(device_type='cuda'):
                preds = self.model(dynamic_inputs, static_inputs, categorical_inputs)
                loss = self.criterion(preds, targets)
                #print(f"Val: Batch {batch_idx + 1} Loss: {loss:.6f}")
                print(f"Stage: val, Step {deload_step} Batch {batch_idx + 1} Loss: {loss:.6f}")

            # Accumulate batch loss into total validation loss
            val_loss += loss.item()

            #self.cleanup(inputs, targets, preds, loss)
        if not self.is_single_dataset:  # Only deload if multiple datasets
            self.data_pipeline.deload_data(step=deload_step, stage="val")
        return total_samples, val_loss

    def cleanup(self, inputs, targets, preds, loss):
        del inputs, targets, preds, loss
        torch.cuda.empty_cache()
        

    def val_loop(self) -> float:
        self.logger.info("Starting Validation")
        total_samples = 0  # Total number of samples processed in the epoch
        val_loss = 0.0  # Total loss for the epoch
        with torch.no_grad():
            for deload_step in range(len(self.data_pipeline.deload_steps)):
                self.logger.info(f"Processing {deload_step + 1}/{len(self.data_pipeline.deload_steps)}")
                
        
                total_samples, val_loss = self.model_val_forward(deload_step, total_samples, val_loss)
                self.logger.info(f"Val {deload_step} Loss: {val_loss:.6f} total_samples: {total_samples}")
                
        
        # Normalize total validation loss by the number of samples
        val_loss /= total_samples
        self.val_losses.append(val_loss)
        return val_loss

    def train_model(self):
        """
        Trains the model with early stopping and mixed precision.
        - Supports multiple GPUs with DataParallel.
        - Saves the best model based on validation loss.
        """
        if torch.cuda.device_count() > 1:
            self.logger.info(f"Using {torch.cuda.device_count()} GPUs for training")
            self.model = torch.nn.DataParallel(self.model)

        self.model.to(self.device)  # Move model to the appropriate device
        for epoch in range(self.num_training_epochs):
            start_time = time.time()
            #self.logger.info(f"Initial Learning Rate: {self.optimizer.param_groups[0]['lr']}")

            train_loss = self.train_loop()
            val_loss = self.val_loop()

            # Early stopping checks
            should_stop, is_best_model = self.early_stopping.check_early_stopping(val_loss,train_loss, epoch, model=self.model)
            if is_best_model:
                self.best_loss = val_loss
                self.logger.info(f"Saving best model with val loss: {val_loss:.6f}")
                #torch.save(self.model.state_dict(), self.best_model_path)  # Save only model weights
                torch.save(self.model, self.best_model_path)
            if should_stop:
                self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

            # Logging
            end_time = time.time()
            self.logger.info(
                f'Epoch {epoch + 1}/{self.num_training_epochs}, '
                f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, '
                f'LR: {self.optimizer.param_groups[0]["lr"]:.2e}, '
                f'Time: {end_time - start_time:.2f}s, '
                f'EarlySC: {self.early_stopping.counter}/{self.early_stopping.patience}, '
                f'opt_step: {self.num_total_optimization_steps[-1]}, '
                f'grad_norm: {self.total_norm:.4f}', time_stamp=False
            )
            ## if the grad_norm is less than 0.01, or inf, or nan, break the train
            if self.total_norm < 0.01 or self.total_norm == float('inf') or self.total_norm != self.total_norm:
                self.logger.info(f"Gradient norm is {self.total_norm}, training is stopped")
                break
            # Plot diagnostics after each epoch
            plot_loss_over_epochs(self.losses, self.val_losses, self.report_path, self.total_norms)
            self.plot_learning_rate_optimization_steps()
        # Final plot for visualization
        plot_loss_over_epochs(self.losses, self.val_losses, self.report_path, self.total_norms)
        self.plot_learning_rate_optimization_steps()
        return self.model_name
