import torch
import matplotlib.pyplot as plt
from PIL import Image
import os
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
from GeoCNN.utils import LoggerSetup
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from PIL import Image
from matplotlib.animation import FFMpegWriter
import imageio
import cv2  # OpenCV for video writing



class InferenceDataset(Dataset):
    def __init__(self, keys_path, h5_path, num_time_steps, group_name, stage):
        self.keys = np.load(keys_path, allow_pickle=True)
        self.h5_file = h5py.File(h5_path, 'r')  # Keep file open
        self.features = self.h5_file[f"{group_name}/{stage}/features/data"]
        self.targets = self.h5_file[f"{group_name}/{stage}/targets/data"]
        self.num_time_steps = num_time_steps

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        keys = self.keys[idx]
        features = self.features[idx, :self.num_time_steps, :, :, :]
        targets = self.targets[idx, :self.num_time_steps, :, :, :]
        return keys, features, targets

    def close(self):
        self.h5_file.close()


class FullInference:
    def __init__(self, config, device=None):
        self.config = config

        self.device = device if device is not None else torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
        self.model_name = config["model_name"]
        self.extracted_data_path = config["extracted_data_path"]
        self.database_path = config["database_path"]
        self.report_path = config["report_path"]
        self.logger = LoggerSetup(self.report_path, verbose=True, rewrite=True)
        self.logger.setup_logger("FullInference")
        self.num_time_steps = config["num_time_steps"]
        self.batch_window = config.get("batch_window", 128)
        self.target_name = "ET"
        self.model = self.get_model()
        

    def get_model(self):
        if torch.cuda.is_available():
            torch.cuda.set_device(self.device)
        else:
            self.logger.warning("CUDA is not available. Running on CPU.")
            self.device = torch.device('cpu')

        model_path = os.path.join(self.report_path, self.model_name, "best_model.pth")
        self.logger.info(f"Loading model from {model_path}")

        model = torch.load(model_path, map_location=self.device)
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        model.to(self.device)
        model.eval()  # Set model to evaluation mode once
        return model

    def get_reference_data(self):
        with h5py.File(self.database_path, 'r') as f:
            base_matrix = f['DEM_250m'][:]
            mask = f['BaseRaster_250m'][:]

        # Directly create base_tensor with desired shape
        base_tensor_shape = (self.num_time_steps,) + base_matrix.shape
        base_tensor = np.zeros(base_tensor_shape, dtype=base_matrix.dtype)
        self.logger.info(f"Size of base tensor: {base_tensor.shape}")
        return base_tensor, mask

    def load_data_loader(self, stage, batch_size=8, batch_window=128):
        group_name = f"{self.target_name}_batch_window_{batch_window}"
        keys_path = os.path.join(self.extracted_data_path, f"{group_name}_{stage}_keys.npy")
        h5_path = os.path.join(self.extracted_data_path, "GeoCNN_data.h5")

        dataset = InferenceDataset(keys_path, h5_path=h5_path, num_time_steps=self.num_time_steps, group_name=group_name, stage=stage)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(stage == 'train'),
            num_workers=4,
            pin_memory=True,
        )

    def evaluate_data_loader(self, data_loader, stage, start_t, end_t, infered_data, infered_data_target):
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (keys_batch, features_batch, targets_batch) in enumerate(data_loader):
                self.logger.info(f"Processing {stage} batch {batch_idx} for time slice {start_t} to {end_t}")
                
                features_batch[features_batch == self.config["no_value"]] = self.config["new_no_value"]
                features_batch = features_batch[:, start_t:end_t].to(self.device, non_blocking=True)
                dynamic_features = features_batch[:, :, 0:9, :, :]    
                static_features = features_batch[:, :, 9:9+16, :, :]
                categorical_features = features_batch[:, :, 9+16:, :, :]
                outputs = self.model(dynamic_features, static_features, categorical_features)
                outputs = outputs.squeeze(2).squeeze(1).cpu().numpy()
                targets_batch = targets_batch[:, start_t:end_t].squeeze(1).numpy()

                keys_array = np.array([list(map(int, str(k).split('_'))) if isinstance(k, (str, bytes)) else k for k in keys_batch])
                min_x, max_x, min_y, max_y = keys_array.T

                for i in range(len(keys_batch)):
                    infered_data[start_t:end_t, min_x[i]:max_x[i], min_y[i]:max_y[i]] = outputs[i]
                    infered_data_target[start_t:end_t, min_x[i]:max_x[i], min_y[i]:max_y[i]] = targets_batch[i].squeeze(1)

        return infered_data, infered_data_target


    def get_data(self, batch_size=8):
        infered_data, mask = self.get_reference_data()
        infered_data_target = np.zeros_like(infered_data)

        total_time_steps = infered_data.shape[0]
        split_time = self.config.get("split_time", total_time_steps)
        stages = ["train", "val", "test"]

 
        for start_t in range(0, total_time_steps, split_time):
            end_t = min(start_t + split_time, total_time_steps)
            self.logger.info(f"Processing time slice {start_t} to {end_t}")

            for stage in stages:
                data_loader = self.load_data_loader(stage, batch_size, self.batch_window)
                self.logger.info(f"Number of batches in {stage}: {len(data_loader)}")

                
                infered_data, infered_data_target = self.evaluate_data_loader(data_loader, stage, start_t, end_t, infered_data, infered_data_target)

        # Apply mask and set negative values to zero
        infered_data[:, mask == -999] = 0
        infered_data_target[:, mask == -999] = 0
        np.maximum(infered_data, 0, out=infered_data)  # In-place operation

        return infered_data, infered_data_target

    def save_data(self, infered_data, target_feature_, output_h5_dir):
        self.logger.info(f"Saving data to {output_h5_dir}")
        if os.path.exists(os.path.join(output_h5_dir, "comparison.h5")):
            self.logger.warning("Comparison file already exists. Skipping saving.")
            return  # Skip saving if file already exists     
        with h5py.File(os.path.join(output_h5_dir, "comparison.h5"), 'w') as f:
            f.create_dataset("predicted", data=infered_data, dtype='float32', compression="gzip", compression_opts=9)
            f.create_dataset("target", data=target_feature_, dtype='float32', compression="gzip", compression_opts=9)
            f.create_dataset("difference", data=infered_data - target_feature_, compression="gzip", compression_opts=9)

            # Optimize NSE calculation
            residual = infered_data - target_feature_
            numerator = np.sum(residual ** 2, axis=0)
            denominator = np.sum((target_feature_ - np.mean(target_feature_, axis=0)) ** 2, axis=0)
            with np.errstate(divide='ignore', invalid='ignore'):
                nse = 1 - np.divide(numerator, denominator, where=denominator != 0)
                nse[denominator == 0] = np.nan
            f.create_dataset("NSE", data=nse, dtype='float32', compression="gzip", compression_opts=9)

            mean_nse = np.nanmean(nse)
            f.create_dataset("mean_NSE", data=mean_nse, dtype='float32')
            f.create_dataset("model_name", data=self.model_name)

            self.logger.info(f"Mean NSE: {mean_nse}")
            self.logger.info("Data saved successfully.")    




    def create_comparison_gif_and_mp4(
        self,
        infered_data,
        target_data,
        output_gif_path,
        output_mp4_path,
        duration=500,
        num_bins=20,
        dpi=150
    ):
        """
        Create both a GIF and an MP4 video comparing target vs. predicted data,
        along with the residual error, using a single color bar.

        Parameters
        ----------
        infered_data : np.ndarray
            The predicted data array (shape: [T, H, W]).
        target_data : np.ndarray
            The ground truth data array (same shape as infered_data).
        output_gif_path : str
            Filepath to save the resulting GIF.
        output_mp4_path : str
            Filepath to save the resulting MP4 video.
        duration : int
            Duration (in milliseconds) per frame for the GIF.
        num_bins : int
            Number of discrete color bins for the Spectral_r colormap.
        dpi : int
            Dots per inch for final figures.
        """
        assert infered_data.shape == target_data.shape, (
            "Predicted and target data must have the same shape"
        )

        # Mask invalid values
        mask_invalid = target_data <= 0
        infered_data_masked = np.ma.masked_where(mask_invalid, infered_data)
        target_data_masked = np.ma.masked_where(mask_invalid, target_data)

        # Calculate residual error
        residual_error = target_data - infered_data
        residual_error_masked = np.ma.masked_where(mask_invalid, residual_error)

        # Load additional masking from HDF5
        with h5py.File(self.database_path, 'r') as f:
            mask = f['BaseRaster_250m'][:]
        # Convert -999 to 0 and everything else to 1
        mask = np.where(mask == -999, 0, 1)

        # Extend the mask over time
        mask = np.expand_dims(mask, axis=0)
        mask = np.repeat(mask, infered_data.shape[0], axis=0)

        # Apply the final mask
        infered_data_masked = np.ma.masked_where(mask == 0, infered_data_masked)
        target_data_masked = np.ma.masked_where(mask == 0, target_data_masked)
        residual_error_masked = np.ma.masked_where(mask == 0, residual_error_masked)

        # Set fixed color range
        global_vmin, global_vmax = 0, 1
        cmap_fixed = plt.get_cmap('Spectral_r', num_bins)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=dpi)

        # Initial images
        im1 = axes[0].imshow(
            target_data_masked[0],
            cmap=cmap_fixed,
            vmin=global_vmin,
            vmax=global_vmax
        )
        axes[0].set_title('Target - Time step 1', pad=5)
        axes[0].axis('off')

        im2 = axes[1].imshow(
            infered_data_masked[0],
            cmap=cmap_fixed,
            vmin=global_vmin,
            vmax=global_vmax
        )
        axes[1].set_title('Predicted - Time step 1', pad=5)
        axes[1].axis('off')

        im3 = axes[2].imshow(
            residual_error_masked[0],
            cmap=cmap_fixed,
            vmin=global_vmin,
            vmax=global_vmax
        )
        axes[2].set_title('Residual Error - Time step 1', pad=5)
        axes[2].axis('off')

        # Single colorbar
        cbar = fig.colorbar(im1, ax=axes.ravel().tolist(),
                            orientation='horizontal',
                            fraction=0.05, pad=0.05)
        cbar.set_label('Value Scale (0 to 1)')

        # List to store each rendered frame as a PIL image (for GIF)
        frames = []

        # List to store frames for MP4
        video_frames = []

        for t in range(infered_data.shape[0]):
            # Update data for each subplot
            im1.set_data(target_data_masked[t])
            im2.set_data(infered_data_masked[t])
            im3.set_data(residual_error_masked[t])

            axes[0].set_title(f'Target - Time step {t + 1}', pad=5)
            axes[1].set_title(f'Predicted - Time step {t + 1}', pad=5)
            axes[2].set_title(f'Residual Error - Time step {t + 1}', pad=5)

            # Draw the updated figure on the canvas
            fig.canvas.draw()

            # Convert canvas to a NumPy array
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            w, h = fig.canvas.get_width_height()
            img = img.reshape((h, w, 3))

            # Save this frame to our frames list (for GIF)
            frames.append(Image.fromarray(img))

            # Save frame for MP4 (Convert RGB to BGR for OpenCV)
            video_frames.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        # Ensure output directories exist
        os.makedirs(os.path.dirname(output_gif_path), exist_ok=True)
        os.makedirs(os.path.dirname(output_mp4_path), exist_ok=True)

        # Create and save GIF using imageio
        imageio.mimsave(output_gif_path, frames, duration=duration/1000)
        self.logger.info(f"GIF saved successfully to {output_gif_path}")

        # Create and save MP4 using OpenCV
        height, width, layers = video_frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec
        fps = 1000 / duration if duration > 0 else 2
        video = cv2.VideoWriter(output_mp4_path, fourcc, fps, (width, height))

        for frame in video_frames:
            video.write(frame)

        video.release()
        self.logger.info(f"MP4 saved successfully to {output_mp4_path}")

        plt.close(fig)

    def create_comparison_gif(self, infered_data, target_data, output_gif_path='comparison.gif', duration=500, num_bins=20, dpi=150):
        """
        Create a GIF to compare target vs. predicted data, along with the residual error, using a single color bar.
        """
        assert infered_data.shape == target_data.shape, "Predicted and target data must have the same shape"

        # Mask invalid values
        mask_invalid = target_data <= 0
        infered_data_masked = np.ma.masked_where(mask_invalid, infered_data)
        target_data_masked = np.ma.masked_where(mask_invalid, target_data)

        # Calculate residual error and mask
        residual_error = target_data - infered_data
        residual_error_masked = np.ma.masked_where(mask_invalid, residual_error)

        # Apply additional masking using BaseRaster_250m
        with h5py.File(self.database_path, 'r') as f:
            mask = f['BaseRaster_250m'][:]
        mask = np.where(mask == -999, 0, 1)

        # Extend the mask over time
        mask = np.expand_dims(mask, axis=0)
        mask = np.repeat(mask, infered_data.shape[0], axis=0)

        infered_data_masked = np.ma.masked_where(mask == 0, infered_data_masked)
        target_data_masked = np.ma.masked_where(mask == 0, target_data_masked)
        residual_error_masked = np.ma.masked_where(mask == 0, residual_error_masked)

        # Set fixed color range
        global_vmin, global_vmax = 0, 1
        cmap_fixed = plt.get_cmap('Spectral_r', num_bins)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.set_dpi(dpi)

        im1 = axes[0].imshow(target_data_masked[0], cmap=cmap_fixed, vmin=global_vmin, vmax=global_vmax)
        axes[0].set_title('Target', pad=5)
        axes[0].axis('off')

        im2 = axes[1].imshow(infered_data_masked[0], cmap=cmap_fixed, vmin=global_vmin, vmax=global_vmax)
        axes[1].set_title('Predicted', pad=5)
        axes[1].axis('off')

        im3 = axes[2].imshow(residual_error_masked[0], cmap=cmap_fixed, vmin=global_vmin, vmax=global_vmax)
        axes[2].set_title('Residual Error', pad=5)
        axes[2].axis('off')

        # Add a single color bar
        cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), orientation='horizontal', fraction=0.05, pad=0.05)
        cbar.set_label('Value Scale (0 to 1)')

        frames = []

        for t in range(infered_data.shape[0]):
            im1.set_data(target_data_masked[t])
            im2.set_data(infered_data_masked[t])
            im3.set_data(residual_error_masked[t])

            axes[0].set_title(f'Target - Time step {t + 1}', pad=5)
            axes[1].set_title(f'Predicted - Time step {t + 1}', pad=5)
            axes[2].set_title(f'Residual Error - Time step {t + 1}', pad=5)

            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(Image.fromarray(img))

        os.makedirs(os.path.dirname(output_gif_path), exist_ok=True)
        frames[0].save(output_gif_path, save_all=True, append_images=frames[1:], duration=duration, loop=0)
        plt.close(fig)
        self.logger.info(f"GIF saved successfully to {output_gif_path}")

def full_inference_helper(model_name, target_name)  -> None:
    config = {
        "model_name": model_name,
        "extracted_data_path": '/home/rafieiva/MyDataBase/codebase/GeoCNN/ml_data/',
        "database_path": '/data/MyDataBase/HydroGeoDataset_ML_250.h5',
        "report_path": '/data/MyDataBase/out/VisionSystem/report',
        "num_time_steps": 60,
        "split_time": 30,
        "batch_window": 128,
        "no_value": -999,
        "new_no_value": 1e-6,
        target_name: target_name
    }

    if f"bw256" in config["model_name"]:
        config["batch_window"] = 256
    if f"bw128" in config["model_name"]:
        config["batch_window"] = 128
    if f"bw64" in config["model_name"]:
        config["batch_window"] = 64
    if f"bw32" in config["model_name"]:
        config["batch_window"] = 32
        
    full_inference_ = FullInference(config)
    
    infered_data, target_feature_ = full_inference_.get_data(batch_size=10)
    ### assert not all infered zero
    assert not np.all(infered_data == 0), "All zero output"
    print(f"range of output: {infered_data.min()} to {infered_data.max()}")
    output_h5_dir = os.path.join(config['report_path'], model_name)
    #full_inference_.save_data(infered_data, target_feature_, output_h5_dir)
    # Create the comparison GIF
    full_inference_.create_comparison_gif_and_mp4(infered_data, target_feature_,
                                                   output_gif_path=f'/data/MyDataBase/out/VisionSystem/report/{config["model_name"]}/{config["model_name"]}_target_vs_predicted.gif',
                                                    output_mp4_path=f'/data/MyDataBase/out/VisionSystem/report/{config["model_name"]}/{config["model_name"]}_target_vs_predicted.mp4',
                                                   duration=500)
    #(infered_data, target_feature_, output_gif_path=f'/data/MyDataBase/out/VisionSystem/report/{config["model_name"]}/{config["model_name"]}_target_vs_predicted.gif', duration=500)


# Example of usage after running inference:
if __name__ == "__main__":


    full_inference_helper("ET_CNNTransformerRegressor_v8_bs28_bw64_lr0_0001_wd0_01_epochs1000_nh8_nl6_fe4_embs1024_dropout0_39", 
                          "ET")

    