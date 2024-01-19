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
rotate = False
beam = True
platform_length = 900
platform_width = 750
division_horizant = 2
division_vertic = 1
tape_width = 8
unit_length = int(platform_length / division_horizant)
unit_width = int(platform_width / division_vertic)
offset_horizant = 10
offset_vertic = 80
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
    "gamma": 1.4,
    # Parameter for the calculation of the extrusion width. Unit: g/mm^3
    "delta": 25.9,
    "tool_number": 0,
    # Tool number of the extruder. Expected to be an integer
    "feedrate": 1800,
    # Feedrate of the extruder in mm/min. Expected to be an integer
    "in_file_path": "/home/yhe/Documents/amworkflow_restruct/examples/RandomPoints/RandomPoints.csv",
    # Path to the input file
    "fixed_feedrate": False,
    "rotate": rotate,
    "density": 2200
    # density of the material in kg/m^3
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


def create_cube_gcode():
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
                current_directory,
                f"/home/yuxiang/Documents/BAM/amworkflow/beam700x150x150x10.csv",
            )
        file_path = os.path.join(
            output_directory,
            f"cube_{infill_type}_150x150x150x{line_width}_P{serial_num}.gcode",
        )
        if beam:
            file_path = os.path.join(
                output_directory,
                "beam_{infill_type}_150x150x150x{line_width}_P{serial_num}.gcode",
            )
        params["offset_from_origin"] = grid[i * division_horizant + j] + offset
        params["line_width"] = line_width
        gcd = GcodeFromPoints(**params)
        gcd.create(data_path, file_path)
        target_bbox[i * division_horizant + j] = np.array([gcd.length, gcd.width])


def create_beam_gcode(infill: str):
    # for i in range(division_horizant):
    for i in range(division_vertic):
        honeycomb_data_path = os.path.join(
            current_directory, "beam_honeycomb_700x150x150x11.4.csv"
        )
        zigzag_data_path = os.path.join(
            current_directory, "beam_zigzag_700x150x150x12.csv"
        )

        if infill == "honeycomb":
            data_path = honeycomb_data_path
            params["line_width"] = 11.4
        else:
            data_path = zigzag_data_path
            params["line_width"] = 12
        file_path = os.path.join(
            output_directory,
            f"beam_{infill}_700x150x150x{params['line_width']}_P{i+1}.gcode",
        )
        params["offset_from_origin"] = grid[i] + offset
        params["kappa"] = 1124.4
        gcd = GcodeFromPoints(**params)
        gcd.create(data_path, file_path)
        target_bbox[i] = np.array([gcd.length, gcd.width])
        target_stdpt[i] = np.array(gcd.btmlftpt)


print(target_stdpt)


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
        # rect_start_x = target_stdpt[i][0] + pt[0]
        # rect_start_y = target_bbox[i][1] + pt[1]
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


create_beam_gcode("zigzag")
plot(grid, platform_length, platform_width)
