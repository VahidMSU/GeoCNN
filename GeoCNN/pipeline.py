import numpy as np
import h5py
import time
from dataclasses import asdict
import os

from GeoCNN.utils import (
    get_swatplus_names,
    LoggerSetup, replace_no_value, 
    chunk_sequences, create_dataloader,
      )
from threading import Thread
from typing import Dict, Any
from multiprocessing import Process, Queue, Event
from GeoCNN.HyperparameterConfig import HyperparameterConfig


def swat_data_loader_helper(config, NAME=None, stage=None):

    """
    Load random non-overlapping chunks from the dataset.
    Open the HDF5 file within each process to avoid resource sharing issues.
    """

    path = config['swatplus_output_path']
    target_name = config['target_array']
    static, dynamic, categorical, target = [], [], [], []
    
    # Define valid range
    with h5py.File(path, 'r') as f:
        total_samples = f[f'{NAME}/dynamic'].shape[0]  # Get the total number of samples

    # Define indices for each stage
    if stage == "train":
        selected_indices = [i for i in range(total_samples) if i not in range(0, total_samples, 4) and i not in range(5, total_samples, 15)]
    elif stage == "val":
        selected_indices = list(range(0, total_samples, 4))  # e.g., 0, 10, 20, ...
    elif stage == "test":
        selected_indices = list(range(5, total_samples, 15))  # e.g., 5, 20, 35, ...
    else:
        raise ValueError(f"Unknown stage: {stage}")

    # Ensure indices are valid
    selected_indices = [i for i in selected_indices if i < total_samples]

    with h5py.File(path, 'r') as f:
        # Load data using the valid indices
        dynamic.append(f[f'{NAME}/dynamic'][selected_indices])
        static.append(f[f'{NAME}/static'][selected_indices])
        categorical.append(f[f'{NAME}/categorical'][selected_indices]) 
        target.append(f[f'{NAME}/target_{target_name}'][selected_indices])

    # Concatenate results
    dynamic_tensor = np.concatenate(dynamic, axis=0)
    static_tensor = np.concatenate(static, axis=0)
    categorical_tensor = np.concatenate(categorical, axis=0)
    target_tensor = np.concatenate(target, axis=0)

    return dynamic_tensor, static_tensor, categorical_tensor, target_tensor


def data_loader_helper(config, deload_step=None, stage=None):

    if config['DataSet']== "HydroGeoDataset":
        from GeoCNN.hydrogeodata_generator import GeoTemporalDataLoader
        
        loader = GeoTemporalDataLoader(config, chunk_num=deload_step)
        return loader.process(stage)
    
    elif config['DataSet']== "SWATplus_output":
        
        return swat_data_loader_helper(config, deload_step, stage)  

class DataPipeline:
    def __init__(self, config, logger=None):
        self.logger = logger or LoggerSetup(config["report_path"], verbose=False, rewrite=True).setup_logger("DataPipeline")
        self.config = config
        self.queue = Queue(maxsize=config["data_pipeline_queue_size"])
        self.stop_event = Event()
        self.current_model_index = 0
        self.deload_steps = self.get_names()
        self.preloading = self.config['preloading']
        self.no_value = self.config["no_value"] 
        self.new_no_value = self.config["new_no_value"]
        self.seq_len = self.config["seq_len"]
        self.batch_size = self.config["batch_size"]
        self.active_data = None

        
        if self.preloading:
            self.logger.info("Preloading data for SWATPlus models.")
            self.preload_process = Process(target=self.preload_data)
            self.preload_process.start()
            self.logger.info("Preloading process started.")
        else:
            self.active_data = None

    def preload_data(self):
        """
        Multiprocessing-based preloading with per-process HDF5 file handling.
        """
        while not self.stop_event.is_set():
            if self.queue.qsize() < self.queue._maxsize:
                self.logger.info(f"Queue size before preload: {self.queue.qsize()}")
                try:
                    chunk_data = self.load_chunk(self.current_model_index)
                    if chunk_data is not None:
                        self.queue.put(chunk_data)
                        self.logger.info(f"Preloaded model {self.current_model_index} into the queue.")
                    self.current_model_index = (self.current_model_index + 1) % len(self.deload_steps)
                    self.logger.info(f"Queue size after preload: {self.queue.qsize()}, current index: {self.current_model_index}")
                except Exception as e:
                    self.logger.error(f"Error during preloading: {e}")
            time.sleep(0.1)

    def stop_preloading(self):
        """
        Stop the preloading process safely.
        """
        if self.preloading:
            self.stop_event.set()
            self.preload_process.join()
            self.logger.info("Preloading process stopped.")

    def get_names(self):
        
        """
        Retrieve the list of SWATPlus model names from the HDF5 file.
        """
        
        if self.config['DataSet'] == "SWATplus_output":
            
            return get_swatplus_names(self.config)
            
        elif self.config['DataSet'] == "HydroGeoDataset":
            return ["HydroGeoDataset"]

    def load_chunk(self, chunk_num):
        """
        Load and preprocess a specific chunk of data.
        """
        #print("chunk_num", chunk_num)
        #print("len(self.deload_steps)", self.deload_steps)
        if chunk_num >= len(self.deload_steps):
            return None

        deload_step = self.deload_steps[chunk_num]
        self.logger.info(f"loading deload_step {deload_step}")
        train_dynamic, train_static, train_categorical, train_target = data_loader_helper(self.config, deload_step=deload_step, stage="train")
        val_dynamic, val_static, val_categorical, val_target = data_loader_helper(self.config, deload_step=deload_step, stage="val")
        test_dynamic, test_static, test_categorical, test_target = data_loader_helper(self.config, deload_step=deload_step, stage="test")
        self.logger.info(f"train_dynamic shape: {train_dynamic.shape}, train_static shape: {train_static.shape}, train_categorical shape: {train_categorical.shape}, train_target shape: {train_target.shape}")
        self.logger.info(f"val_dynamic shape: {val_dynamic.shape}, val_static shape: {val_static.shape}, val_categorical shape: {val_categorical.shape}, val_target shape: {val_target.shape}")
        self.logger.info(f"test_dynamic shape: {test_dynamic.shape}, test_static shape: {test_static.shape}, test_categorical shape: {test_categorical.shape}, test_target shape: {test_target.shape}")

        train_dynamic = replace_no_value(train_dynamic, self.no_value, self.new_no_value)
        train_static = replace_no_value(train_static, self.no_value, self.new_no_value)
        train_categorical = replace_no_value(train_categorical, self.no_value, 0)
        train_target = replace_no_value(train_target, self.no_value, self.new_no_value)
        
        val_dynamic = replace_no_value(val_dynamic, self.no_value, self.new_no_value)
        val_static = replace_no_value(val_static, self.no_value, self.new_no_value)
        val_categorical = replace_no_value(val_categorical, self.no_value, 0)
        val_target = replace_no_value(val_target, self.no_value, self.new_no_value)

        test_dynamic = replace_no_value(test_dynamic, self.no_value, self.new_no_value)
        test_static = replace_no_value(test_static, self.no_value, self.new_no_value)
        test_categorical = replace_no_value(test_categorical, self.no_value, 0)
        test_target = replace_no_value(test_target, self.no_value, self.new_no_value)
 
        # Example Usage
        data_dict = {
            "dynamic": train_dynamic,
            "static": train_static,
            "target": train_target,
            "categorical": train_categorical
        }

        chunked_train_data = chunk_sequences(data_dict, seq_len=self.seq_len)

        data_dict_val = {
            "dynamic": val_dynamic,
            "static": val_static,
            "target": val_target,
            "categorical": val_categorical
        }

        chunked_val_data = chunk_sequences(data_dict_val, seq_len=self.seq_len)


        data_dict_test = {
            "dynamic": test_dynamic,
            "static": test_static,
            "target": test_target,
            "categorical": test_categorical
        }

        chunked_test_data = chunk_sequences(data_dict_test, seq_len=self.seq_len)


        return {
            "model_name": deload_step,
            "train_loader": create_dataloader(chunked_train_data["dynamic"], chunked_train_data["static"], chunked_train_data["categorical"], chunked_train_data["target"], batch_size=self.batch_size, shuffle=True), 
            "val_loader": create_dataloader(chunked_val_data["dynamic"], chunked_val_data["static"], chunked_val_data["categorical"], chunked_val_data["target"], batch_size=self.batch_size, shuffle=False),
            "test_loader": create_dataloader(chunked_test_data["dynamic"], chunked_test_data["static"], chunked_test_data["categorical"], chunked_test_data["target"], batch_size=1, shuffle=False),
            "total_time_steps": train_dynamic.shape[1], 
            "num_dynamic_channels": train_dynamic.shape[2], 
            "num_static_channels": train_static.shape[2],
            "num_categorical_channels": train_categorical.shape[2],
            "num_channels": train_dynamic.shape[2] + train_static.shape[2] + train_categorical.shape[2],
            "steps_per_load": (
                (train_dynamic.shape[0] + self.batch_size - 1) // self.batch_size
            ),


        }

    def update_chunk(self, chunk_num, stage="train"):
        """
        Load and preprocess a specific chunk of data for a given stage.
        
        Args:
            chunk_num (int): Chunk number to load
            stage (str): Data stage - 'train', 'val', or 'test'
        
        Returns:
            dict: Loaded data configuration or None if chunk not available
        """
        if chunk_num >= len(self.deload_steps):
            return None


        deload_step = self.deload_steps[chunk_num]
        self.logger.info(f"Updating chunk {deload_step} for {stage} stage.")
        dynamic, static, categorical, target = data_loader_helper(self.config, deload_step=deload_step, stage=stage)
        dynamic = replace_no_value(dynamic, self.no_value, self.new_no_value)
        static = replace_no_value(static, self.no_value, self.new_no_value)
        categorical = replace_no_value(categorical, self.no_value, 0)
        target = replace_no_value(target, self.no_value, self.new_no_value)
        
        data_dict = {
            "dynamic": dynamic,
            "static": static,
            "target": target,
            "categorical": categorical
        }

        chunked_data = chunk_sequences(data_dict, seq_len=self.seq_len)

        dynamic = chunked_data["dynamic"]
        static = chunked_data["static"]
        categorical = chunked_data["categorical"]
        target = chunked_data["target"]

        batch_size = self.batch_size if stage != "test" else 1
        shuffle = stage == "train"
        
        if dynamic.size == 0 or static.size == 0 or categorical.size == 0 or target.size == 0:
            self.logger.warning(f"Chunk {deload_step} for {stage} stage is empty. Skipping.")
            return None
        
        return {
            "model_name": deload_step,
            f"{stage}_loader": create_dataloader(dynamic, static, categorical, target, batch_size=batch_size, shuffle=shuffle), 
            "total_time_steps": dynamic.shape[1],
            "num_dynamic_channels": dynamic.shape[2],
            "num_static_channels": static.shape[2],
            "num_categorical_channels": categorical.shape[2],   
            "steps_per_load": (dynamic.shape[0] + batch_size - 1) // batch_size
        }
    def update_loaders(self, step=None, stage=None):
        if step is None:
            step = 0  # Default to the first step
            print("step is None")

        if not self.preloading:
            self.logger.info(f"loading data for step {step}")
            print(f"reloading is {self.preloading}")
            self.active_data = self.load_chunk(step)  # Pass the step argument here
            if stage == "train":
                self.active_data.update(self.update_chunk(step, stage="train"))
            elif stage == "val":
                self.active_data.update(self.update_chunk(step, stage="val"))
            elif stage == "test":
                self.active_data.update(self.update_chunk(step, stage="test"))
            return

        while self.queue.qsize() < self.config["data_pipeline_queue_size"]//3 and self.config['DataSet'] == "SWATplus_output":
            print(f"required queue size is {self.config['data_pipeline_queue_size']//2} and current queue size is {self.queue.qsize()}")
            self.logger.info(f"Waiting for queue to fill. Current size: {self.queue.qsize()}")
            time.sleep(10)

        if self.queue.empty() and self.config['DataSet'] == "SWATplus_output":
            raise StopIteration("No more data chunks available.")
        
        chunk_data = self.queue.get()
        self.active_data = chunk_data
        self.logger.info(f"Loaded model: {self.active_data['model_name']}")


    

    def deload_data(self, step, stage=None):
        """
        Clear the current active data and update the next model if available.
        """

        if not self.preloading:
            self.active_data = None
            self.update_loaders(step, stage)
            return
        else:

            if self.queue.qsize() >= self.config["data_pipeline_queue_size"]//4:
                self.logger.info(f"Queue has sufficient data ({self.queue.qsize()} items). Skipping immediate update.")
                self.active_data = self.queue.get()
                return

            # Otherwise, update in the background
            self.logger.info(f"Queue size is low ({self.queue.qsize()} items). Triggering background update.")
            update_thread = Thread(target=self.update_loaders, args=(step, stage))
            update_thread.start()

            # Use available data for current step
            if not self.queue.empty():
                self.active_data = self.queue.get()
            else:
                raise StopIteration("No more data chunks available.")

            

    def get_active_data_loaders(self):
        """
        Return the active data loaders and metadata.
        """
        if not self.active_data:
            raise ValueError("No active data is loaded. Call update_loaders first.")
        return (
            self.active_data["train_loader"],
            self.active_data["val_loader"],
            self.active_data["test_loader"],
            self.active_data["total_time_steps"],
            self.active_data["num_channels"],
            self.active_data["steps_per_load"],
        )

    def get_active_train_loaders(self):
        """
        Return the active data loaders and metadata.
        """
        if not self.active_data:
            raise ValueError("No active data is loaded. Call update_loaders first.")
        return (
            self.active_data["train_loader"]
        )
    
    def get_active_val_loaders(self):
        """
        Return the active data loaders and metadata.
        """
        if not self.active_data:
            raise ValueError("No active data is loaded. Call update_loaders first.")
        return (
            self.active_data["val_loader"]

        )
    def get_active_test_loaders(self):
        """
        Return the active data loaders and metadata.
        """
        if not self.active_data:
            raise ValueError("No active data is loaded. Call update_loaders first.")
        return (
            self.active_data["test_loader"]
        )
    
def run_training_pipeline(config: Dict[str, Any]) -> None:
    """
    Testing data pipeline with dummy training loop.

    Args:
        config (dict): Configuration dictionary for the pipeline.
    """
    logger = LoggerSetup(config["report_path"], verbose=False, rewrite=True).setup_logger("TrainingPipeline")
    data_pipeline = DataPipeline(config)
    ### update loaders
    data_pipeline.update_loaders(step=0)
    ### Training loop
    epoches = 3 # dummy value
    for epoch in range(epoches):
        ## training
        _start_time = time.time()   
        for deload_step in range(len(data_pipeline.deload_steps)):
            train_loader = data_pipeline.get_active_train_loaders()
            start_time = time.time()
            for i, (dynamic, static, categorical, target) in enumerate(train_loader):
                logger.info(f"Epoch {epoch}, Stage: Training, Step {deload_step}, Batch {i}: Inputs {dynamic.shape}, Targets {target.shape}")
                ## define training loop
                #time.sleep(1) # dummy model 
            data_pipeline.deload_data(step=deload_step, stage="train")
            logger.info(f"Training step {deload_step} completed in {time.time() - start_time:.2f} seconds.")  
        logger.info(f"Epoch {epoch} completed in {time.time() - _start_time:.2f} seconds.")
        ## validation
        _start_time = time.time()
        for deload_step in range(len(data_pipeline.deload_steps)):
            val_loader = data_pipeline.get_active_val_loaders()
            start_time = time.time()
            for i, (dynamic, static, categorical, target) in enumerate(val_loader):
                logger.info(f"Epoch {epoch}, Stage: Validation, Step {deload_step}, Batch {i}: Inputs {dynamic.shape}, Targets {target.shape}")
                ## define validation loop
                #time.sleep(1) # dummy model
            data_pipeline.deload_data(step=deload_step, stage="val")
            logger.info(f"Validation step {deload_step} completed in {time.time() - start_time:.2f} seconds.")
    ## testing
    for deload_step in range(len(data_pipeline.deload_steps)):
        test_loader = data_pipeline.get_active_test_loaders()
        for i, (dynamic, static, categorical, target) in enumerate(test_loader):
            logger.info(f"Evaluation, Step {deload_step}, Batch {i}: Inputs {dynamic.shape}, Targets {target.shape}")
            ## define testing loop
            #time.sleep(1)
        data_pipeline.deload_data(step=deload_step, stage="test")



    data_pipeline.stop_preloading()
    logger.info("Training pipeline completed.")


if __name__ == "__main__":
    config = HyperparameterConfig(
        report_path=os.getcwd(),
        DataSet="SWATplus_output",
        target_array="perc"
    )

    print(asdict(config))

    config = asdict(config)


    run_training_pipeline(config)
