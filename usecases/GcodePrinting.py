import os
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from amworkflow.gcode.gcode import GcodeFromPoints, GcodeMultiplier

rotate = True

params = {  # geometry parameters
    "layer_num": 50,
    # Number of printed layers. expected to be an integer
    "layer_height": 3,
    # Layer height in mm
    "line_width": 10,
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
    "kappa": 192.410,
    # Parameter for the calculation of the extrusion length
    "gamma": 1.816667,
    # Parameter for the calculation of the extrusion width. Unit: g/mm^3
    "delta": 1.16,
    "tool_number": 0,
    # Tool number of the extruder. Expected to be an integer
    "feedrate": 1800,
    # Feedrate of the extruder in mm/min. Expected to be an integer
    "in_file_path": "/Users/yuxianghe/Documents/BAM/amworkflow_restructure/usecases/cube_honeycomb_150x150x150x10.csv",
    # Path to the input file
    "fixed_feedrate": False,
    "rotate": rotate,
    "density": 2200
    # density of the material in kg/m^3
}

multiGCD = GcodeMultiplier(2, 1, 1020, 800, params)
multiGCD.create(auto_balance=False, dist_horizont=250, dist_vertic=0)
multiGCD.visualize()
# multiGCD.create(auto_balance=False, dist_horizont=250, dist_vertic=100)
