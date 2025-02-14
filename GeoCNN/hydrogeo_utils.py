

import h5py
import numpy as np
import torch





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




def get_categorical_data(f, month, year, min_x, max_x, min_y, max_y, no_value, size, batch_features, year_idx, num_climate_vars, num_modis_vars, numerical_var, categorical_var, unique_cat_vars, logger=None):
    valid_batch = True

    if len(categorical_var) > 0:
        for feature_idx, var in enumerate(categorical_var):
            data = f[var][min_x:max_x, min_y:max_y]
            unique_values = np.unique(data)
            i = 0
            for unique_value in unique_values[unique_values != no_value]:
                if unique_value in unique_cat_vars:
                    if data.shape != (size, size):
                        if logger:
                            logger.warning(f"Skipping batch due to invalid shape: {data.shape}")
                        valid_batch = False
                        break

                    batch_features[year_idx * 12 + month - 1, 
                                   num_climate_vars + num_modis_vars + len(numerical_var) + feature_idx + i, 
                                   :, :] = torch.tensor(np.where(data == unique_value, 1, 0))
                    i += 1
    
    return valid_batch, batch_features

def get_numerical_data(f, year, month, min_x, max_x, min_y, max_y, local_mask, no_value, size, batch_features, year_idx, num_climate_vars, num_modis_vars, numerical_var):
    valid_batch = True

    for feature_idx, var in enumerate(numerical_var):
        data = f[var][:][min_x:max_x, min_y:max_y]
        data = np.array(data, dtype=np.float32)  # Ensure data is a NumPy array with float type

        if f[var].attrs.get(f'global_min_{var}') is None:
            global_min = np.nanmin(data[data != no_value])
            global_max = np.nanmax(data[data != no_value])
            f[var].attrs[f'global_min_{var}'] = global_min
            f[var].attrs[f'global_max_{var}'] = global_max
        else:
            global_min = f[var].attrs[f'global_min_{var}']
            global_max = f[var].attrs[f'global_max_{var}']

        data[local_mask == no_value] = no_value
        data = np.where(data != no_value, (data - global_min) / (global_max - global_min), no_value)

        assert np.min(data[data!=no_value]) >= 0 and np.max(data[data!=no_value]) <= 1, f"Data not between 0 and 1: {var}, {np.min(data[data!=no_value])}, {np.max(data[data!=no_value])}"

        if data.shape != (size, size):
            valid_batch = False
            break

        batch_features[year_idx * 12 + month - 1, num_climate_vars + num_modis_vars + feature_idx, :, :] = torch.tensor(data)

    return valid_batch, batch_features

def get_prism_data(f, year, month, min_x, max_x, min_y, max_y, local_mask, no_value, size, batch_features, year_idx):
    valid_batch = True

    for var_idx, var in enumerate(["ppt", "tmax", "tmin"]):
        var_add = f"PRISM_monthly/{var}_{year}_{month}"
        global_min = f["PRISM_monthly"].attrs[f'global_min_{var}']
        global_max = f["PRISM_monthly"].attrs[f'global_max_{var}']
        var_data = f[var_add][min_x:max_x, min_y:max_y]
        var_data[local_mask == no_value] = no_value   

        var_data = np.where(var_data != no_value, (var_data - global_min) / (global_max - global_min), no_value)

        if var_data.shape != (size, size):
            valid_batch = False
            break
        
        batch_features[year_idx * 12 + month - 1, var_idx, :, :] = torch.tensor(var_data)
    
    return valid_batch, batch_features
def get_modis_as_feature(f, year, month, min_x, max_x, min_y, max_y, local_mask, no_value, batch_features, year_idx, num_climate_vars):
    valid_batch = True

    for modis_idx, mod in enumerate(['MOD09GQ_sur_refl_b01', 'MOD09GQ_sur_refl_b02', 'MOD13Q1_EVI', 'MOD13Q1_NDVI', 'MOD15A2H_Fpar_500m', 'MOD15A2H_Lai_500m']):
        modis_keys = f[mod].keys()
        global_min = f[mod].attrs['global_min']
        global_max = f[mod].attrs['global_max']
        
        var_MOD = f"{mod}_{year}{month:02d}"
        var_MOD = [x for x in modis_keys if var_MOD in x][0]
        
        data_batch = f[f"{mod}/{var_MOD}"][min_x:max_x, min_y:max_y]
        data_batch = np.array(data_batch, dtype=np.float32)  # Ensure data is a NumPy array with float type 
        
        ### making sure the data is within the mask
        data_batch[local_mask == no_value] = no_value
        
        ## scaling the data
        data_batch = np.where(data_batch != no_value, (data_batch - global_min) / (global_max - global_min), no_value)
        
        batch_features[year_idx * 12 + month - 1, num_climate_vars + modis_idx, :, :] = torch.tensor(data_batch)
    
    return valid_batch, batch_features


#from GeoCNN.SWATplus_dataloader import swat_data_loader_helper
def get_modis_as_target(f, min_x, max_x, min_y, max_y, year, month, local_mask, no_value, modis_group = "MOD16A2_ET"):

    modis_group_keys = f[modis_group].keys()
    
    def get_modis_mon_batch(year, month):
        """
        Retrieve MODIS ET data for a specific month with error handling.
        
        Args:
            year (int): Year of the data
            month (int): Month of the data
        
        Returns:
            tuple: Data batch and valid values
        """
        try:
            data_month = f"{modis_group}_{year}{month:02d}"
            matching_keys = [x for x in modis_group_keys if data_month in x]
            
            if not matching_keys:
                raise ValueError(f"No data found for {data_month}")
            
            data_batch = f[f"{modis_group}/{matching_keys[0]}"][min_x:max_x, min_y:max_y]
            data_batch = np.array(data_batch, dtype=np.float32)
            data_batch[local_mask == no_value] = no_value
            
            valid_values = data_batch[data_batch != no_value]
            return data_batch, valid_values
        
        except Exception as e:
            print(f"Error retrieving data for {year}-{month}: {e}")
            return np.full((max_x-min_x, max_y-min_y), no_value, dtype=np.float32), np.array([])

    # Determine previous, current, and next month
    prev_year, prev_month = (year-1, 12) if month == 1 else (year, month-1)
    next_year, next_month = (year+1, 1) if month == 12 else (year, month+1)

    # Retrieve data for adjacent months
    pre_mon, valid_pre_mon = get_modis_mon_batch(prev_year, prev_month)
    current_mon, valid_values = get_modis_mon_batch(year, month)
    next_mon, valid_next_mon = get_modis_mon_batch(next_year, next_month)

    # Create mask for regions with consecutive no_value
    new_mask = np.where(
        (local_mask != no_value) & 
        (current_mon == no_value) & 
        (next_mon == no_value) & 
        (pre_mon == no_value), 
        1, 0
    )

    # Handle data filling strategies
    if np.all(new_mask == 1):
        current_mon[new_mask == 1] = 0.0
    else:
        # Fill no_value regions with interpolated or averaged values
        current_mon[new_mask == 1] = no_value
        pre_mon[new_mask == 1] = no_value
        next_mon[new_mask == 1] = no_value

        # Compute means for interpolation
        pre_mon_mean = np.mean(valid_pre_mon) if valid_pre_mon.size > 0 else 0.0
        next_mon_mean = np.mean(valid_next_mon) if valid_next_mon.size > 0 else 0.0
        current_mon_mean = np.mean(valid_values) if valid_values.size > 0 else 0.0

        # Interpolation strategy
        interpolated_mean = np.mean([pre_mon_mean, current_mon_mean, next_mon_mean])

        # Fill no_value regions
        mask_condition = (current_mon == no_value) & (local_mask != no_value)
        current_mon[mask_condition] = interpolated_mean

    # Global min-max scaling
    global_min = f[modis_group].attrs['global_min']
    global_max = f[modis_group].attrs['global_max']

    current_mon = np.where(
        current_mon != no_value, 
        (current_mon - global_min) / (global_max - global_min), 
        no_value
    )

    return torch.tensor(current_mon)
