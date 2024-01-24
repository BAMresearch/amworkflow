import os
from pathlib import Path

import numpy as np

from amworkflow.src.gcode.gcode import GcodeFromPoints

honeycomb_path = (
    "/home/yuxiang/Documents/BAM/amworkflow/cube_honeycomb_150x150x150x10.csv"
)
zigzag_path = "/home/yuxiang/Documents/BAM/amworkflow/cube_zigzag_150x150x150x10.csv"

platform_length = 1188
platform_width = 772
unit_length = int(platform_length / 4)
unit_width = int(platform_width / 3)
offset = [5, 80 + 5]


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

mypath = "/home/yuxiang/Documents/BAM/amworkflow/cube_honeycomb_150x150x150x10.csv"
file_gcode = (
    "/home/yuxiang/Documents/BAM/amworkflow/cube_honeycomb_150x150x150x10_P2.gcode"
)
serial_num = 0
for i in range(0, 4):
    for j in range(0, 3):
        serial_num += 1
        if serial_num <= 6:
            infill_type = "honeycomb"
            line_width = 10
        else:
            infill_type = "zigzag"
            line_width = 11.3
        data_path = f"/home/yuxiang/Documents/BAM/amworkflow/cube_{infill_type}_150x150x150x{line_width}.csv"
        file_path = f"/home/yuxiang/Documents/BAM/amworkflow/cube_{infill_type}_150x150x150x{line_width}_P{serial_num}.gcode"
        x_coord = i * unit_length
        y_coord = j * unit_width
        coordinate = np.array([x_coord, y_coord]) + np.array(offset)
        params["offset_from_origin"] = coordinate
        params["line_width"] = line_width
        gcd = GcodeFromPoints(**params)
        gcd.create(data_path, file_path)
