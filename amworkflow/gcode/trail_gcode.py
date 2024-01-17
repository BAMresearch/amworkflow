import os
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from amworkflow.gcode.gcode import GcodeFromPoints

honeycomb_path = (
    "/home/yuxiang/Documents/BAM/amworkflow/cube_honeycomb_150x150x150x10.csv"
)
zigzag_path = "/home/yuxiang/Documents/BAM/amworkflow/cube_zigzag_150x150x150x10.csv"

platform_length = 1188
platform_width = 772
division_horizant = 4
division_vertic = 3
tape_width = 8
unit_length = int(platform_length / division_horizant)
unit_width = int(platform_width / division_vertic)
offset_horizant = 5
offset_vertic = 5
offset = (
    np.array([offset_horizant, offset_vertic]) + np.array([tape_width, tape_width]) / 2
)
total_num = division_vertic * division_horizant

grid = np.zeros((total_num, 2))
for i in range(0, division_vertic):
    for j in range(0, division_horizant):
        grid[i * division_horizant + j] = np.array([j * unit_length, i * unit_width])

print(grid)
params = {  # geometry parameters
    "layer_num": 50,
    # Number of printed layers. expected to be an integer
    "layer_height": 3,
    # Layer height in mm
    "line_width": 11,
    # Line width in mm
    "offset_from_origin": [50, 50],
    # Offset from origin in mm
    "unit": "mm",
    # Unit of the geometry
    "standard": "ConcretePrinter",
    # Standard of the printer firmware
    "coordinate_system": "absolute",
    # Coordinate system of the printer firmware
    "nozzle_diameter": 8.1,
    # Diameter of the nozzle in mm
    "kappa": 181.5954,
    # Parameter for the calculation of the extrusion length
    "gamma": 0.0043,
    # Parameter for the calculation of the extrusion width
    "delta": 25.9,
    "tool_number": 0,
    # Tool number of the extruder. Expected to be an integer
    "feedrate": 1800,
    # Feedrate of the extruder in mm/min. Expected to be an integer
    "in_file_path": "/home/yhe/Documents/amworkflow_restruct/examples/RandomPoints/RandomPoints.csv",
    # Path to the input file
    "fixed_feedrate": True,
}

# mypath = "/home/yuxiang/Documents/BAM/amworkflow/cube_honeycomb_150x150x150x10.csv"
# file_gcode = (
#     "/home/yuxiang/Documents/BAM/amworkflow/cube_honeycomb_150x150x150x10_P2.gcode"
# )

current_directory = os.getcwd()
current_directory = os.path.dirname(os.path.dirname(current_directory))
output_directory = os.path.join(current_directory, "output")
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

target_bbox = np.zeros((total_num, 2))

serial_num = 0
for i in range(0, division_vertic):
    for j in range(0, division_horizant):
        serial_num += 1
        if serial_num <= 6:
            infill_type = "honeycomb"
            line_width = 10
        else:
            infill_type = "zigzag"
            line_width = 11.3
        data_path = os.path.join(
            current_directory, f"cube_{infill_type}_150x150x150x{line_width}.csv"
        )
        file_path = os.path.join(
            output_directory,
            f"cube_{infill_type}_150x150x150x{line_width}_P{serial_num}.gcode",
        )
        params["offset_from_origin"] = grid[i * division_horizant + j] + offset
        params["line_width"] = line_width
        gcd = GcodeFromPoints(**params)
        gcd.create(data_path, file_path)
        target_bbox[i * division_horizant + j] = np.array([gcd.length, gcd.width])


def plot(coordinates, total_length, total_width):
    # Add boundary points
    coordinates_plus = np.vstack(
        (
            np.array([0, 0]),
            np.array([0, total_width]),
            np.array([total_length, total_width]),
            np.array([total_length, 0]),
            coordinates,
        )
    )
    # Extract x and y coordinates
    x_coords = coordinates_plus[:, 0]
    y_coords = coordinates_plus[:, 1]

    # Plotting
    plt.figure(figsize=(8, 8))
    plt.scatter(x_coords, y_coords, marker="o")  # Plot points

    # Annotating each point with its coordinate
    for x, y in coordinates:
        plt.text(x, y, f"({x},{y})", ha="right", va="bottom", fontsize=8)

    # Optional: Draw lines to form a grid
    for x in np.unique(x_coords):
        plt.axvline(x, color="lightgrey", linestyle="--")
    for y in np.unique(y_coords):
        plt.axhline(y, color="lightgrey", linestyle="--")

    for i, pt in enumerate(coordinates):
        pt += offset
        rect_start_x = pt[0]
        rect_start_y = pt[1]
        length = target_bbox[i][0]
        width = target_bbox[i][1]
        rect = patches.Rectangle(
            (rect_start_x, rect_start_y),
            length,
            width,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        plt.gca().add_patch(rect)

    # plt.grid(True)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Grid Plot with Coordinates")
    plt.show()


plot(grid, platform_length, platform_width)
