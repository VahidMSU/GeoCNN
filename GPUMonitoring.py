import time
import csv
import matplotlib.pyplot as plt
from pynvml import *

# Initialize NVML
nvmlInit()
device_count = nvmlDeviceGetCount()

# Open a CSV file to save the logs
with open("gpu_usage_log.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["timestamp", "gpu_id", "gpu_utilization", "memory_used", "memory_total"])

    # Set up live plot
    plt.ion()  # Interactive mode on
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Dictionary to store utilization and memory usage for each GPU
    utilization_data = {i: [] for i in range(device_count)}
    memory_data = {i: [] for i in range(device_count)}
    time_data = []

    # Define color map for different GPUs
    colors = plt.cm.tab10(range(device_count))

    try:
        while True:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            time_data.append(timestamp)

            # Collect data for each GPU
            for i in range(device_count):
                handle = nvmlDeviceGetHandleByIndex(i)
                utilization = nvmlDeviceGetUtilizationRates(handle)
                memory = nvmlDeviceGetMemoryInfo(handle)

                # Write data to CSV
                writer.writerow([
                    timestamp,
                    i,
                    utilization.gpu,
                    memory.used / (1024 ** 2),  # Convert to MB
                    memory.total / (1024 ** 2)  # Convert to MB
                ])
                file.flush()

                # Append data for plotting
                utilization_data[i].append(utilization.gpu)
                memory_data[i].append(memory.used / (1024 ** 2))

                # Keep the last 100 entries for smoother real-time plotting
                if len(time_data) > 100:
                    time_data.pop(0)
                    utilization_data[i].pop(0)
                    memory_data[i].pop(0)

            # Clear and plot GPU Utilization
            ax1.clear()
            ax1.set_title("GPU Utilization (%)")
            ax1.set_xlabel("Time")
            ax1.set_ylabel("Utilization (%)")
            for i in range(device_count):
                ax1.plot(time_data, utilization_data[i], label=f"GPU {i}", color=colors[i])
            ax1.legend(loc="upper left")
            ax1.tick_params(axis='x', rotation=45)

            # Clear and plot Memory Usage
            ax2.clear()
            ax2.set_title("GPU Memory Usage (MB)")
            ax2.set_xlabel("Time")
            ax2.set_ylabel("Memory Used (MB)")
            for i in range(device_count):
                ax2.plot(time_data, memory_data[i], label=f"GPU {i}", color=colors[i])
            ax2.legend(loc="upper left")
            ax2.tick_params(axis='x', rotation=45)

            # Update the plot
            plt.tight_layout()
            plt.pause(1)  # Pause to update the plot every second
            plt.savefig("gpu_usage_plot.png")
            # Wait for 1 second before the next reading
            time.sleep(1)

    except KeyboardInterrupt:
        print("Monitoring stopped.")

    finally:
        # Close NVML
        nvmlShutdown()
