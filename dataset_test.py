import h5py
import numpy as np  
def plot_ET_distribution():
    hydrogeodataset_path = "/data/MyDataBase/HydroGeoDataset_ML_250.h5"

    with h5py.File(hydrogeodataset_path, "r") as f:
        # List all groups
        print(f"Keys: {f['MODIS_ET'].keys()}")
        keys = list(f['MODIS_ET'].keys())
        all_data = []
        for key in keys:
            print(f"Key: {key}")
            data = f[f'MODIS_ET/{key}'][:]
            all_data.append(data)

        all_data = np.concatenate(all_data, axis=0)
        all_data = np.where(all_data <0, np.nan, all_data)

        ## max and min values
        print(f"Max: {np.nanmax(all_data)}")
        print(f"Min: {np.nanmin(all_data)}")
        print(f"Mean: {np.nanmean(all_data)}")
        print(f"STD: {np.nanstd(all_data)}")
        ## plot distribution
        import matplotlib.pyplot as plt 
        all_data = all_data[~np.isnan(all_data)]
        all_data = all_data[all_data > 0]
        ### scale between zero and 1
        all_data = (all_data - np.min(all_data)) / (np.max(all_data) - np.min(all_data))

        plt.hist(all_data.flatten(), bins=100)
        plt.title("MODIS ET Distribution")
        plt.xlabel("ET")
        plt.ylabel("Frequency")
        plt.grid( linestyle='--', linewidth=0.5)
        plt.savefig("MODIS_ET.png", dpi=300)    

#plot_ET_distribution()


def plot_SWAT_groundwater_recharge():
    SWATplus_output_path = "/data/MyDataBase/out/SWATplus_output/CentralSWAT_data.h5"

    with h5py.File(SWATplus_output_path, "r") as f:
        keys = list(f.keys())
        print(f"Keys: {keys}")
        for key in keys:
            print(f"{key}-categorical shape: {f[f'{key}/categorical'].shape}")
            print(f"{key}-static shape: {f[f'{key}/static'].shape}")
            print(f"{key}-dynamic shape: {f[f'{key}/dynamic'].shape}")

        ### only keys with less than 5 character
        keys = [key for key in keys if len(key) < 5]
        all_tensors = []
        for key in keys:
            data = f[key]["target_perc"][:]
            all_tensors.append(data)

        all_tensors = np.concatenate(all_tensors, axis=0)
        all_tensors = np.where(all_tensors <0, np.nan, all_tensors)

        ## plot distribution
        import matplotlib.pyplot as plt
        all_tensors = all_tensors[~np.isnan(all_tensors)]
        all_tensors = all_tensors[all_tensors > 0]

        plt.hist(all_tensors.flatten(), bins=100)
        plt.title("SWAT+ groundwate recharge Distribution")
        plt.xlabel("Scaled groundwater recharge")
        plt.ylabel("Frequency")
        plt.grid( linestyle='--', linewidth=0.5)
        plt.savefig("SWAT_groundwater_recharge.png", dpi=300)

    # The output should be:

plot_SWAT_groundwater_recharge()