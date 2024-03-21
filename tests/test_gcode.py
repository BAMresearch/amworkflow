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
    # Coefficient of rectifying the extrusion length
    "delta": 0.1,
    # Coefficient of rectifying the feedrate, as well as the line width
    "gamma": 1,
    # Coefficient of rectifying the feedrate, as well as the line width
    "tool_number": 0,
    # Tool number of the extruder. Expected to be an integer
    "feedrate": 1800,
    # Feedrate of the extruder in mm/min. Expected to be an integer
}


def test_gcode(tmp_path):
    caller_path = Path(os.path.dirname(__file__))
    file_point = caller_path / "RandomPoints.csv"
    gcd = GcodeFromPoints(**params)
    file_gcode = Path(tmp_path) / "test.gcode"
    gcd.create(file_point, file_gcode)
    assert file_gcode.exists()
