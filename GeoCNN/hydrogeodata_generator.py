import os
import numpy as np
import torch
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
try:
    from GeoCNN.registry import retrieve_features
except ImportError:
    from registry import retrieve_features  
import time
import h5py
#from GeoCNN.SWATplus_dataloader import swat_data_loader_helper

from GeoCNN.utils import LoggerSetup



def extract_prism_snodas(path, prism_path, batch_window, time_steps, number_of_years, years, target_name):
    """ 
    Extracts data from SNODAS and PRISM HDF5 files and combines them into a single tensor.
    """
    prism_batches = []
    prism_batch_keys = []
    # Initialize empty lists to store batches
    snodas_batches = []
    snodas_batch_keys = []  
    target_batches = []
    target_batch_keys = []
    # Open SNODAS data

    

    with h5py.File(path, 'r') as f:

        data_shape = f['250m']["2004"]['melt_rate'].shape  # e.g., (time_steps, height, width)
        height, width = data_shape[1], data_shape[2]
        
        # Calculate the number of batches that fit in each dimension
        num_batches_y = height // batch_window
        num_batches_x = width // batch_window

        # Loop through each batch window
        for _ in range(num_batches_y):
            for _ in range(num_batches_x):
                min_x = _ * batch_window
                max_x = (_ + 1) * batch_window
                min_y = _ * batch_window
                max_y = (_ + 1) * batch_window
                
                # Create a tensor for each batch
                snodas_tensor = torch.zeros(1, number_of_years*time_steps, 4, batch_window, batch_window)
                target_tensor = torch.zeros(1, number_of_years*time_steps, 1, batch_window, batch_window)   
                
                # Extract data for each variable within the batch window
                for i, var in enumerate(['melt_rate', 'snow_accumulation', 'snow_layer_thickness', 'snow_water_equivalent', 'snowpack_sublimation_rate']):

                    #print(f'extraction of {var} for batch {by} {bx}')
                    global_min = f['250m'].attrs[f'{var}_global_min_mm']
                    global_max = f['250m'].attrs[f'{var}_global_max_mm']
                    for j, year in enumerate(years):
                        data = f[f'250m/{year}/{var}'][:time_steps,
                                                    min_y:max_y,
                                                    min_x:max_x]
                        converter = f[f'250m/{year}/{var}'].attrs['converters']
                        if var == target_name:
                            data = np.where(data == 55537, -999, (data*converter - global_min) / (global_max - global_min))
                            snodas_tensor[0, j*time_steps:(j+1)*time_steps, i, :, :] = torch.tensor(data)
                        else:
                            data = np.where(data == 55537, -999, (data*converter - global_min) / (global_max - global_min))
                            target_tensor[0, j*time_steps:(j+1)*time_steps, 0, :, :] = torch.tensor(data)
                            
                        
                # Append the tensor for this batch to the list of batches
                snodas_batches.append(snodas_tensor)
                snodas_batch_keys.append(f"{min_x}_{max_x}_{min_y}_{max_y}")
                target_batches.append(target_tensor)
                target_batch_keys.append(f"{min_x}_{max_x}_{min_y}_{max_y}")

    # Open PRISM data
    with h5py.File(prism_path, 'r') as f:
        for _ in range(num_batches_y):
            for _ in range(num_batches_x):
                # Create a tensor for each batch
                prism_tensor = torch.zeros(1, number_of_years*time_steps, 3, batch_window, batch_window)
                
                # Extract data for each variable within the batch window
                for i, var in enumerate(['ppt', "tmax", "tmin"]):
                    global_min = f[var].attrs[f'{var}_global_min']
                    global_max = f[var].attrs[f'{var}_global_max']
                    for j, year in enumerate(years):
                        #print(f'extraction of {var} for year {year} for batch {_} {_}')
                        data = f[f'{var}/{year}/data'][:time_steps,
                                                        min_y:max_y,
                                                        min_x:max_x]
                        data = np.where(data == -999, -999, (data - global_min) / (global_max - global_min))
                        # Replace invalid values (-999) with -0.01
                        
                        # Add the data to the tensor
                        prism_tensor[0, j*time_steps:(j+1)*time_steps, i, :, :] = torch.tensor(data)
                
                # Append the tensor for this batch to the list of batches
                prism_batches.append(prism_tensor)

    # Convert the lists of batches to tensors
    snodas_tensor = torch.cat(snodas_batches, dim=0)
    prism_tensor = torch.cat(prism_batches, dim=0)
    target_tensor = torch.cat(target_batches, dim=0)

    ### combine into one tensor
    all_features = torch.cat([snodas_tensor, prism_tensor], dim=2)
    # Print the shape of the combined tensor
    print(f"shape of the feature tensor: {all_features.shape}")
    print(f"shape of the target tensor: {target_tensor.shape}")
    # Return the combined tensor


    ## split the data into training and testing sets
    train_size = int(0.7 * len(all_features))
    val_size = int(0.2 * len(all_features))
    test_size = len(all_features) - train_size - val_size

    all_train_features, all_val_features, all_test_features = torch.split(all_features, [train_size, val_size, test_size])
    all_train_targets, all_val_targets, all_test_targets = torch.split(target_tensor, [train_size, val_size, test_size])
    all_train_keys, all_val_keys, all_test_keys = snodas_batch_keys[:train_size], snodas_batch_keys[train_size:train_size + val_size], snodas_batch_keys[train_size + val_size:]
    print("===================Features==========================")
    print(f"shape of the training feature tensor: {all_train_features.shape}")
    print(f"shape of the validation feature tensor: {all_val_features.shape}")
    print(f"shape of the testing feature tensor: {all_test_features.shape}")
    print("===================Targets==========================")
    print(f"shape of the training target tensor: {all_train_targets.shape}")
    print(f"shape of the validation target tensor: {all_val_targets.shape}")
    print(f"shape of the testing target tensor: {all_test_targets.shape}")


    with h5py.File("ml_data/GeoCNN_data.h5", 'w') as f:
        group_name = f"{target_name}_batch_window_{batch_window}"
        f.create_group(f"{group_name}")
        f[f"{group_name}"].create_group("train")
        f[f"{group_name}"].create_group("val")
        f[f"{group_name}"].create_group("test")
        f[f"{group_name}/train"].create_group("features")
        f[f"{group_name}/train"].create_group("targets")
        f[f"{group_name}/val"].create_group("features")
        f[f"{group_name}/val"].create_group("targets")
        f[f"{group_name}/test"].create_group("features")
        f[f"{group_name}/test"].create_group("targets")
        f[f"{group_name}/train/features"].create_dataset("data", data=all_train_features.numpy(), compression="gzip", compression_opts=5)
        f[f"{group_name}/train/targets"].create_dataset("data", data=all_train_targets.numpy(), compression="gzip", compression_opts=5)
        f[f"{group_name}/val/features"].create_dataset("data", data=all_val_features.numpy(), compression="gzip", compression_opts=5)
        f[f"{group_name}/val/targets"].create_dataset("data", data=all_val_targets.numpy(), compression="gzip", compression_opts=5)
        f[f"{group_name}/test/features"].create_dataset("data", data=all_test_features.numpy(), compression="gzip", compression_opts=5)
        f[f"{group_name}/test/targets"].create_dataset("data", data=all_test_targets.numpy(), compression="gzip", compression_opts=5)
        np.save(f"ml_data/{group_name}_train_keys.npy", all_train_keys)
        np.save(f"ml_data/{group_name}_val_keys.npy", all_val_keys)
        np.save(f"ml_data/{group_name}_test_keys.npy", all_test_keys)

    return all_train_features, all_train_targets, all_val_features, all_val_targets, all_test_features, all_test_targets

class GeoTemporalDataLoader:
    def __init__(self, config, chunk_num=None):
        self.config = config
        self.numerical_var = retrieve_features("default_feature_retriever", "numerical", 250)
        self.categorical_var = retrieve_features("default_feature_retriever", "categorical", 250)
        self.all_feature_names = self.numerical_var + self.categorical_var
        self.target_name = config['target_array']
        self.years_of_target = [int(config['start_year']), int(config['end_year'])]
        self.hydrogeodataset_path = config['hydrogeodataset_path']
        self.size = config['batch_window']
        self.overwrite = config['overwrite']
        self.no_value = config['no_value']  
        self.batch_window = config['batch_window']  
        self.no_value_distribution = config['no_value_distribution']
        self.extracted_dataset_path = config['extracted_dataset_path']
        assert os.path.exists(self.hydrogeodataset_path), "H5 file not found"
        self.chunk_num = chunk_num  
        self.logger = LoggerSetup("logs", rewrite=True)
        self.logger.setup_logger("HydroGeoDataset")

    def data_loader_worker(self, f_path, unique_values_chunk):
        with h5py.File(f_path, 'r') as f:
            batch_region = f[f"{self.size}_{self.size}_batch_size"][:]
            mask = f['BaseRaster_250m'][:]
            return self.data_loader(f, batch_region, mask, unique_values_chunk)

    def unique_categorical_values (self, f,  unique_values):
        land_cover_types = {
            11: "WATR",
            12: "WATR",
            21: "URLD",
            22: "URMD",
            23: "URHD",
            24: "UIDU",
            31: "SWRN",
            41: "FRSD",
            42: "FRSE",
            43: "FRST",
            52: "RNGB",
            71: "RNGE",
            81: "HAY",
            82: "AGRR",
            90: "WETF",
            95: "WETL"
        }
        ### read unique values in the categorical valiables
        unique_values = set()
        for var in self.categorical_var:
            if var == "landuse_250m":
                for uniq in np.unique(f[var][:]):
                    ## only waters and wetlands
                    if uniq in [11, 12, 90, 95]:
                        unique_values.add(uniq)
                        print(f"Adding {land_cover_types[uniq]}")
            else:
                unique_values.append(np.unique(f[var][:]))
    
        unique_values = list(unique_values)
        print(f"number of unique values: {unique_values}")
        return unique_values
    

    def get_modis_product_batch(self, f, min_x, max_x, min_y, max_y, year, month, modis_product_keys, local_mask, modis_group="MOD16A2_ET"):
        pattern = f"{modis_group}_{year}{month:02d}"
        #print(f"%%%%%%%%%%%%%%%%%year: {year}, month: {month}")
        pattern = [x for x in modis_product_keys if pattern in x][0]
        data_batch = f[f"{modis_group}/{pattern}"][min_x:max_x, min_y:max_y]
        data_batch = np.array(data_batch, dtype=np.float32)  # Ensure data is a NumPy array with float type   
        data_batch[local_mask == self.no_value] = self.no_value
        # Ensure there are valid values to compute the mean
        valid_values = data_batch[(data_batch != self.no_value)]
        return data_batch, valid_values
    
    def data_loader(self, f, batch_region, mask, unique_values):
        num_years = self.years_of_target[1] - self.years_of_target[0] + 1
        num_climate_vars = 3  # ppt, tmax, tmin
        num_modis_vars = 6 # 6 modis variables
        unique_cat_vars = self.unique_categorical_values(f, unique_values)
        num_total_features = num_climate_vars + num_modis_vars + len(self.numerical_var) + len(unique_cat_vars)
        self.logger.info(f"Number of numerical variables: {len(self.numerical_var)}, Number of categorical variables: {len(self.categorical_var)}, Number of unique categorical variables: {unique_cat_vars}")
        valid_features = []
        valid_targets = []
        batch_keys = []
        # Add tqdm for progress bar over unique values (batches)
        for batch_idx, value in tqdm(enumerate(unique_values), total=len(unique_values), desc="Loading batches"):
            region = np.where(batch_region == value)
            global_mask = mask
            min_x, max_x = region[0].min(), region[0].max() + 1
            min_y, max_y = region[1].min(), region[1].max() + 1
            local_mask = mask[min_x:max_x, min_y:max_y]

            if (max_x - min_x != self.size) or (max_y - min_y != self.size):
                continue

            batch_features = torch.zeros((num_years * 12, num_total_features, self.size, self.size))
            batch_targets = torch.zeros((num_years * 12, 1, self.size, self.size))

            valid_batch = True

            for year_idx, year in enumerate(range(self.years_of_target[0], self.years_of_target[1] + 1)):
                for month in range(1, 13):
                    if "ET" in self.target_name:
                        modis_group = "MOD16A2_ET"
                    else:
                        raise NotImplementedError(f"Target name {self.target_name} not supported")
                
                    modis_product_keys = f[modis_group].keys()
                    global_min = f[modis_group].attrs['global_min']
                    global_max = f[modis_group].attrs['global_max']

                    
                    #ET_data = self.get_ET_data(f, min_x, max_x, min_y, max_y, year, month, ET_names, local_mask)
                    #### previous month
                    if month != 1:
                        pre_mon, _valid_values = self.get_modis_product_batch(f, min_x, max_x, min_y, max_y, year, month-1, modis_product_keys, local_mask, modis_group)
                    else:
                        pre_mon, _valid_values = self.get_modis_product_batch(f, min_x, max_x, min_y, max_y, year-1, 12, modis_product_keys, local_mask, modis_group)
                    #### current month
                    current_mon, valid_values = self.get_modis_product_batch(f, min_x, max_x, min_y, max_y, year, month, modis_product_keys, local_mask, modis_group)
                    ### new month
                    # If the current month is December, move to January of the next year
                    if month == 12:
                        next_mon, valid_values_ = self.get_modis_product_batch(f, min_x, max_x, min_y, max_y, year + 1, 1, modis_product_keys, local_mask, modis_group)
                    else:
                        next_mon, valid_values_ = self.get_modis_product_batch(f, min_x, max_x, min_y, max_y, year, month + 1, modis_product_keys, local_mask, modis_group)

                    ### if three consecutive months are no_value, then it's lakes and no-values that must be set 0
                    new_mask = np.where((local_mask != self.no_value) & (current_mon == self.no_value) & (next_mon == self.no_value) & (pre_mon == self.no_value), 1, 0)
                    #print(f"#######################New mask: {np.sum(new_mask)}, shape: {new_mask.shape}, local mask shape: {local_mask.shape}")
                    if np.all(new_mask == 1):
                        # If all values are no_value, set the entire region to 0
                        current_mon[new_mask == 1] = 0.0
                    else:
                        # Handle cases with mixed no_value and valid values
                        current_mon[new_mask == 1] = self.no_value  
                        # No need to assign to `valid_values` here
                        pre_mon[new_mask == 1] = self.no_value 
                        next_mon[new_mask == 1] = self.no_value 

                        # If there are still no_values in the current month, replace them with the mean of adjacent months
                        if valid_values.size > 0:
                            _mean_value = np.mean(_valid_values) if _valid_values.size > 0 else 0.0
                            mean_value_ = np.mean(valid_values_) if valid_values_.size > 0 else 0.0
                            mean_value = np.mean(valid_values)
                            mean_value = (mean_value + _mean_value + mean_value_) / 3

                            # Fill remaining no_value in ET_data with the calculated mean
                            current_mon[(current_mon == self.no_value) & (local_mask != self.no_value)] = mean_value
                        else:
                            valid_mask = (pre_mon != self.no_value) & (next_mon != self.no_value)
                            averaged_values = np.where(valid_mask, (pre_mon + next_mon) / 2, self.no_value)

                            # Ensure the shape matches the mask
                            current_mon[(current_mon == self.no_value) & (local_mask != self.no_value)] = averaged_values[
                                (current_mon == self.no_value) & (local_mask != self.no_value)
                            ]

                        #assert np.all(data[data == no_value] == no_value), "No_value not handled correctly!"
                        assert np.all(current_mon[local_mask == self.no_value] == self.no_value), "No_value not handled correctly!"
                        # Scale the data
                        current_mon = np.where(current_mon != self.no_value, (current_mon - global_min) / (global_max - global_min), self.no_value)

                        ### adding the data to the batch
                        batch_targets[year_idx * 12 + month - 1, 0, :, :] = torch.tensor(current_mon)
                    for var_idx, var in enumerate(["ppt", "tmax", "tmin"]):
                        var_add = f"PRISM_monthly/{var}_{year}_{month}"
                        global_min = f["PRISM_monthly"].attrs[f'global_min_{var}']
                        global_max = f["PRISM_monthly"].attrs[f'global_max_{var}']
                        var_data = f[var_add][min_x:max_x, min_y:max_y]
                        var_data[local_mask==self.no_value] = self.no_value   
                        
                        var_data = np.where(var_data != self.no_value, (var_data - global_min) / (global_max - global_min), self.no_value)

                        if var_data.shape != (self.size, self.size):
                            valid_batch = False
                            break

                        if np.sum(var_data == self.no_value) > 0.8 * self.size * self.size:
                            valid_batch = False
                            break
                        batch_features[year_idx * 12 + month - 1, var_idx, :, :] = torch.tensor(var_data)
  
                    if not valid_batch:
                        break

                    ## add other modis data
                    for modis_idx, mod in enumerate(['MOD09GQ_sur_refl_b01', 'MOD09GQ_sur_refl_b02', 'MOD13Q1_EVI', 'MOD13Q1_NDVI', 'MOD15A2H_Fpar_500m', 'MOD15A2H_Lai_500m']):
                        modis_products = f[mod].keys()
                        global_min = f[mod].attrs['global_min']
                        global_max = f[mod].attrs['global_max']
                        var_MOD = f"{mod}_{year}{month:02d}"
                        var_MOD = [x for x in modis_products if var_MOD in x][0]
                        data_batch = f[f"{mod}/{var_MOD}"][min_x:max_x, min_y:max_y]
                        data_batch = np.array(data_batch, dtype=np.float32)  # Ensure data is a NumPy array with float type 
                        ### making sure the data is within the mask
                        data_batch[local_mask == self.no_value] = self.no_value
                        ## scaling the data
                        data_batch = np.where(data_batch != self.no_value, (data_batch - global_min) / (global_max - global_min), self.no_value)
                        batch_features[year_idx * 12 + month - 1, num_climate_vars + modis_idx, :, :] = torch.tensor(data_batch)


                    for feature_idx, var in enumerate(self.numerical_var):
                        data = f[var][:][min_x:max_x, min_y:max_y]
                        data = np.array(data, dtype=np.float32)  # Ensure data is a NumPy array with float type

                        if f[var].attrs.get(f'global_min_{var}') is None:
                            global_min = np.nanmin(data[data != self.no_value])
                            global_max = np.nanmax(data[data != self.no_value])
                            f[var].attrs[f'global_min_{var}'] = global_min
                            f[var].attrs[f'global_max_{var}'] = global_max
                        else:
                            global_min = f[var].attrs[f'global_min_{var}']
                            global_max = f[var].attrs[f'global_max_{var}']

                        data[local_mask == self.no_value] = self.no_value
                        data = np.where(data != self.no_value, (data - global_min) / (global_max - global_min), self.no_value)

                        assert np.min(data[data!=self.no_value]) >= 0 and np.max(data[data!=self.no_value]) <= 1, f"Data not between 0 and 1: {var}, {np.min(data[data!=self.no_value])}, {np.max(data[data!=self.no_value])}"

                        if data.shape != (self.size, self.size):
                            valid_batch = False
                            break

                        batch_features[year_idx * 12 + month - 1, num_climate_vars + num_modis_vars + feature_idx, :, :] = torch.tensor(data)
                    if len(self.categorical_var) > 0:
                        for feature_idx, var in enumerate(self.categorical_var):
                            data = f[var][min_x:max_x, min_y:max_y]
                            #data[local_mask != 1] = self.no_value   
                            unique_values = np.unique(data)
                            i = 0
                            for unique_value in unique_values[unique_values != self.no_value]:
                                if unique_value in unique_cat_vars:
                        
                                    if data.shape != (self.size, self.size):
                                        valid_batch = False
                                        break

                                    batch_features[year_idx * 12 + month - 1, num_climate_vars + num_modis_vars + len(self.numerical_var) + feature_idx + i, :, :] = torch.tensor(np.where(data == unique_value, 1, 0))
                                    i += 1
                    if not valid_batch:
                        break

            if valid_batch:
                valid_features.append(batch_features)
                valid_targets.append(batch_targets)
                batch_key = f"{min_x}_{max_x}_{min_y}_{max_y}"
                batch_keys.append(batch_key)

        if valid_features and valid_targets:
            all_features = torch.stack(valid_features, dim=0)
            all_targets = torch.stack(valid_targets, dim=0)
        else:
            all_features = torch.empty(0)
            all_targets = torch.empty(0)

        return all_features, all_targets, batch_keys

    def load_GeoCNN_data(self, group_name, stage):
        with h5py.File(self.extracted_dataset_path, 'r') as f:
            self.logger.info(f"Loading {stage} data")
            dynamic_features = f[f"{group_name}/{stage}/features/data"][:,:, :9,:,:]
            static_features = f[f"{group_name}/{stage}/features/data"][:,:, 9:9+16,:,:]
            categorical_features = f[f"{group_name}/{stage}/features/data"][:,:, 9+16:,:,:] 
            targets = f[f"{group_name}/{stage}/targets/data"][:]
            self.logger.info(f"Dyanmic features shape: {dynamic_features.shape}, Static features shape: {static_features.shape}, Categorical features shape: {categorical_features.shape}, Targets shape: {targets.shape}") 
        return dynamic_features, static_features, categorical_features, targets

    def extract_split_save_data(self, group_name):
        with h5py.File(self.hydrogeodataset_path, 'r') as f:    
            # Read the batch region and mask
            batch_region = f[f"{self.size}_{self.size}_batch_size"][:]  # Batch regions (self.size x self.size blocks)
            mask = f['BaseRaster_250m'][:]  # Mask for valid regions
            assert self.no_value in mask, "No value not found in mask"

        with h5py.File(self.extracted_dataset_path, 'w') as f:
            self._extracted_from_extract_split_save_data_9(batch_region, group_name, f)
        self.logger.info(f"Saved data to {self.extracted_dataset_path}")

    # TODO Rename this here and in `extract_split_save_data`
    def _extracted_from_extract_split_save_data_9(self, batch_region, group_name, f):
        unique_values = np.unique(batch_region)
        unique_values = unique_values[unique_values != self.no_value]  # Remove no_value entries

        # Split unique values into chunks of 24
        unique_value_chunks = [unique_values[i:i + 6] for i in range(0, len(unique_values), 6)]
        parallel = True
        if parallel:
            # Use ProcessPoolExecutor to load each chunk with one worker
            with ProcessPoolExecutor() as executor:
                results = list(tqdm(executor.map(self.data_loader_worker, [self.hydrogeodataset_path] * len(unique_value_chunks), unique_value_chunks), total=len(unique_value_chunks), desc="Processing batches with workers"))
        else:
            # Load each chunk sequentially
            results = [self.data_loader_worker(self.hydrogeodataset_path, chunk) for chunk in tqdm(unique_value_chunks, desc="Processing batches")]

        # Combine results
        all_features = torch.cat([result[0] for result in results if result[0].size(0) > 0], dim=0)
        all_targets = torch.cat([result[1] for result in results if result[1].size(0) > 0], dim=0)
        all_keys = [key for result in results for key in result[2]]

        # Split after loading all data
        if self.no_value_distribution == "random":
            self.random_split(all_features, all_targets, all_keys, group_name)
        elif self.no_value_distribution == "clean_training":
            self.save_split_clean_training(all_features, all_targets, all_keys, f, group_name)
        else:
            raise NotImplementedError(f"Invalid no_value_distribution: {self.no_value_distribution}")

    def save_split_clean_training(self, all_features, all_targets, all_keys, f, group_name):
        no_value_threshold_1 = 0.25
        no_value_threshold_2 = 0.5
        train_size = int(0.7 * all_features.size(0))
        val_size = int(0.2 * all_features.size(0))

        splits = {
            'train': {'features': [], 'targets': [], 'keys': []},
            'val': {'features': [], 'targets': [], 'keys': []},
            'test': {'features': [], 'targets': [], 'keys': []}
        }

        for i in range(all_targets.shape[0]):
            no_value_ratio = torch.sum(all_targets[i] == self.no_value) / (self.size * self.size * all_targets.size(1))
            if no_value_ratio < no_value_threshold_1 and train_size > 0:
                splits['train']['features'].append(all_features[i:i+1])
                splits['train']['targets'].append(all_targets[i:i+1])
                splits['train']['keys'].append(all_keys[i])
                train_size -= 1
            elif no_value_ratio < no_value_threshold_2 and val_size > 0:
                splits['val']['features'].append(all_features[i:i+1])
                splits['val']['targets'].append(all_targets[i:i+1])
                splits['val']['keys'].append(all_keys[i])
                val_size -= 1
            else:
                splits['test']['features'].append(all_features[i:i+1])
                splits['test']['targets'].append(all_targets[i:i+1])
                splits['test']['keys'].append(all_keys[i])

        # Ensure group structure exists
        self.ensure_group_exists(f, group_name)

        # Save each split
        for stage in ['train', 'val', 'test']:
            stage_group = f[f"{group_name}/{stage}"]
            stage_group.require_group("features")
            stage_group.require_group("targets")

            if splits[stage]['features']:
                features = torch.cat(splits[stage]['features'], dim=0).numpy()
                targets = torch.cat(splits[stage]['targets'], dim=0).numpy()

                stage_group["features"].create_dataset("data", data=features, compression="gzip", compression_opts=5)
                stage_group["targets"].create_dataset("data", data=targets, compression="gzip", compression_opts=5)

                # Save keys as a separate dataset
                np.save(f"ml_data/{group_name}_{stage}_keys.npy", splits[stage]['keys'])

    def random_split(self, all_features, all_targets, all_keys, group_name):

        train_size = int(0.7 * len(all_features))
        val_size = int(0.2 * len(all_features))
        test_size = len(all_features) - train_size - val_size
        all_train_features, all_val_features, all_test_features = torch.split(all_features, [train_size, val_size, test_size])
        all_train_targets, all_val_targets, all_test_targets = torch.split(all_targets, [train_size, val_size, test_size])
        all_train_keys, all_val_keys, all_test_keys = all_keys[:train_size], all_keys[train_size:train_size + val_size], all_keys[train_size + val_size:]
        
        with h5py.File(self.extracted_dataset_path, 'w') as f:
            self.ensure_group_exists(f, group_name)
            for data, stage in zip([all_train_features, all_val_features, all_test_features], ["train", "val", "test"]):
                self.logger.info(f"Saving {stage} data")
                f[f"{group_name}/{stage}/features"].create_dataset("data", data=data.numpy(), compression="gzip", compression_opts=5)
            for data, stage in zip([all_train_targets, all_val_targets, all_test_targets], ["train", "val", "test"]):
                f[f"{group_name}/{stage}/targets"].create_dataset("data", data=data.numpy(), compression="gzip", compression_opts=5)
            for data, stage in zip([all_train_keys, all_val_keys, all_test_keys], ["train", "val", "test"]):
                np.save(f"ml_data/{group_name}_{stage}_keys.npy", data)
        self.logger.info(f"Saved data to {self.extracted_dataset_path}")
        
    def create_groups(self, f, group_name):
        f.create_group(f"{group_name}")
        ## create group for train, val, test
        f[f"{group_name}"].create_group("train")
        f[f"{group_name}"].create_group("val")
        f[f"{group_name}"].create_group("test")
        self.logger.info(f"Created groups for {group_name}")    

    def ensure_group_exists(self, f, group_name):
        if f.get(f"{group_name}") is None:
            self.create_groups(f, group_name)

    def process(self, stage):
        if os.path.exists(self.extracted_dataset_path):
            with h5py.File(self.extracted_dataset_path, 'r') as f:
                keys = list(f.keys())
        else: 
            keys = []
            self.overwrite = True
        print(f"keys: {keys}")
        group_name = f"{self.target_name}_batch_window_{self.batch_window}"

        if group_name not in keys or self.overwrite:
            self.extract_split_save_data(group_name)
        return self.load_GeoCNN_data(group_name, stage)

def plot_single_distribution(data, name, no_value):
    import matplotlib.pyplot as plt
    #data = np.where(data == 0, np.nan, data)
    data = np.where(data == no_value, np.nan, data)
    plt.hist(data.flatten(), bins=100)
    plt.title(f"Distribution of {name}")
    os.makedirs("figs", exist_ok=True)
    plt.savefig(f"figs/{name}.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    
    config = {
        "DataSet" : "HydroGeoDataset",
        "no_value_distribution": "clean_training",
        'method': 'never_seen',
        'overwrite': False,
        'target_array': "ET",
        'hydrogeodataset_path': "/data/MyDataBase/HydroGeoDataset_ML_250.h5",
        "extracted_dataset_path": "ml_data/GeoCNN_data.h5",
        
        'RESOLUTION': 250,
        'no_value': -999,
        'start_year': 2002,
        'end_year': 2021,
        'batch_window': 64
    }
    loader = GeoTemporalDataLoader(config)
    
    dynamic_features, static_features, categorical_features, targets = loader.process(stage="train")
    #for i in range(features.shape[2]):
    #    plot_single_distribution(features[:,:,i,:,:], f"channel_{i}", config['no_value'])    
    #plot_single_distribution(targets, "target", config['no_value'])
    

 #   # Paths to your HDF5 files
 #   target_name = "melt_rate"
 #   path = "/data/MyDataBase/SNODAS.h5"
 #   prism_path = "/data/MyDataBase/PRISM_ML_250m.h5"
 #   batch_window = 128
 #   time_steps = 10
 #   years = [2005]
 #   number_of_years = len(years)
 #   extract_prism_snodas(path, prism_path, batch_window, time_steps, number_of_years, years, target_name)