import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def seconds_to_minutes(seconds):
    # Convert seconds to minutes
    return seconds / 60


def plot_time_aggregate_volume_mass(file_path):
    # Read data from the provided file path
    data = pd.read_csv(file_path)

    # Extracting the necessary columns
    time = data["time"]
    aggregate_volume = data["aggregate volume"]
    mass = data["aggregate mass"]

    # Setup for interception lines every 15 minutes (900 seconds)
    intercept_intervals = np.arange(0, time.max(), 900)

    # Create a figure object
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    # Plotting Time vs Aggregate Volume with intercepts
    axs[0].plot(time, aggregate_volume, "b-o")
    for intercept in intercept_intervals:
        intersect_y = np.interp(intercept, time, aggregate_volume)
        axs[0].axvline(x=intercept, color="gray", linestyle="--")
        axs[0].text(
            intercept,
            intersect_y,
            f"({seconds_to_minutes(intercept):.2f} min, {intersect_y:.2e})",
            verticalalignment="bottom",
            horizontalalignment="right",
        )
    axs[0].set_title("Time vs Aggregate Volume")
    axs[0].set_xlabel("Time (seconds)")
    axs[0].set_ylabel("Aggregate Volume")

    # Plotting Time vs Mass with intercepts
    axs[1].plot(time, mass, "r-s")
    for intercept in intercept_intervals:
        intersect_y = np.interp(intercept, time, mass)
        axs[1].axvline(x=intercept, color="gray", linestyle="--")
        axs[1].text(
            intercept,
            intersect_y,
            f"({seconds_to_minutes(intercept):.2f} min, {intersect_y})",
            verticalalignment="bottom",
            horizontalalignment="right",
        )
    axs[1].set_title("Time vs Mass")
    axs[1].set_xlabel("Time (seconds)")
    axs[1].set_ylabel("Mass")

    plt.tight_layout()

    # Return the figure object instead of showing it
    return fig


def plot_data(slope, intercept):
    x = np.linspace(-10, 10, 100)
    y = slope * x + intercept
    plt.figure(figsize=(8, 4))
    plt.plot(x, y)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Line Plot")
    plt.grid(True)
    plt.close()  # This prevents the plot from being displayed in the notebook
    return plt.gcf()  # Return the figure object


def create_dataframe(num_rows, num_cols):
    data = np.random.rand(num_rows, num_cols)
    df = pd.DataFrame(data, columns=[f"Column {i+1}" for i in range(num_cols)])
    return df


plot_interface = gr.Interface(
    fn=plot_time_aggregate_volume_mass,
    inputs=gr.Textbox(label="File Path"),
    outputs=gr.Plot(),
    title="Line Plot",
    analytics_enabled=False,
)

dataframe_interface = gr.Interface(
    fn=create_dataframe,
    inputs=[
        gr.Slider(1, 10, step=1, value=5, label="Number of Rows"),
        gr.Slider(1, 5, step=1, value=3, label="Number of Columns"),
    ],
    outputs=gr.Dataframe(),
    title="Dataframe",
)

iface = gr.TabbedInterface([plot_interface, dataframe_interface], ["Plot", "Dataframe"])

iface.launch()


# Example of how to use this function:
# plot_time_aggregate_volume_mass(
#     "/Users/yuxianghe/Documents/BAM/amworkflow_restructure/amworkflow/gcode/log_beam_zigzag_700x150x150x12_P1.csv"
# )
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
