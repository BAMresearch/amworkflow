import matplotlib.pyplot as plt
import pandas as pd


def plot_time_aggregate_volume_mass(file_path):
    # Read data from the provided file path
    data = pd.read_csv(file_path)

    # Extracting the necessary columns
    time = data["time"]
    aggregate_volume = data["aggregate volume"]
    mass = data["aggregate mass"]

    # Plotting Time vs Aggregate Volume
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
    plt.plot(time, aggregate_volume, "b-o")  # 'b-o' is a blue line with circle markers
    plt.title("Time vs Aggregate Volume")
    plt.xlabel("Time")
    plt.ylabel("Aggregate Volume")

    # Plotting Time vs Mass
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
    plt.plot(time, mass, "r-s")  # 'r-s' is a red line with square markers
    plt.title("Time vs Mass")
    plt.xlabel("Time")
    plt.ylabel("Mass")

    plt.tight_layout()
    plt.show()


# Example of how to use this function:
plot_time_aggregate_volume_mass(
    "/Users/yuxianghe/Documents/BAM/amworkflow_restructure/amworkflow/gcode/log_beam_zigzag_700x150x150x12_P1.csv"
)
