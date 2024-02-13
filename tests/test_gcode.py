import os
from pathlib import Path

from amworkflow.gcode.gcode import GcodeFromPoints

# define required parameters
params = {  # geometry parameters
    "layer_num": 1,
    # Number of printed layers. expected to be an integer
    "layer_height": 1,
    # Layer height in mm
    "line_width": 1,
    # Line width in mm
    "offset_from_origin": [0, 0],
    # Offset from origin in mm
    "unit": "mm",
    # Unit of the geometry
    "standard": "ConcretePrinter",
    # Standard of the printer firmware
    "coordinate_system": "absolute",
    # Coordinate system of the printer firmware
    "nozzle_diameter": 0.4,
    # Diameter of the nozzle in mm
    "kappa": 1,
    # Parameter for the calculation of the extrusion width
    "tool_number": 0,
    # Tool number of the extruder. Expected to be an integer
    "feedrate": 1800,
    # Feedrate of the extruder in mm/min. Expected to be an integer
    "in_file_path": "/home/yhe/Documents/amworkflow_restruct/examples/RandomPoints/RandomPoints.csv",
    # Path to the input file
}


def test_gcode(tmp_path):
    caller_path = Path(os.path.dirname(__file__))
    file_point = caller_path / "RandomPoints.csv"
    params["in_file_path"] = file_point
    gcd = GcodeFromPoints(**params)
    file_gcode = tmp_path / "RandomPoints.gcode"
    gcd.create(file_point, file_gcode)
    assert file_gcode.exists()
