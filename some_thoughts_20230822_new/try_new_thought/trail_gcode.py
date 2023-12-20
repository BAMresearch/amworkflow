import os
from pathlib import Path

from amworkflow.src.gcode.gcode import GcodeFromPoints

params = {  # geometry parameters
    "layer_num": 50,
    # Number of printed layers. expected to be an integer
    "layer_height": 3,
    # Layer height in mm
    "line_width": 11,
    # Line width in mm
    "offset_from_origin": [0, 0],
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
    "feedrate": 3000,
    # Feedrate of the extruder in mm/min. Expected to be an integer
    "in_file_path": "/home/yhe/Documents/amworkflow_restruct/examples/RandomPoints/RandomPoints.csv",
    # Path to the input file
    "fixed_feedrate": False,
}

mypath = "/Users/yuxianghe/Documents/BAM/amworkflow_restructure/beam700x150x150x10.csv"
file_gcode = (
    "/Users/yuxianghe/Documents/BAM/amworkflow_restructure/beam700x150x150x10.gcode"
)
gcd = GcodeFromPoints(**params)
gcd.create(mypath, file_gcode)
