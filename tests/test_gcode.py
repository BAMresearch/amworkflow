import os
from pathlib import Path
import pytest
import logging

from amworkflow.gcode.gcode import GcodeFromPoints
logging.basicConfig(level=logging.INFO)
# define required parameters
params1 = {  # geometry parameters
    "layer_num": 2,
    # Number of printed layers. expected to be an integer
    "layer_height": 1,
    # Layer height in mm
    "line_width": 1,
    # Line width in mm
    "offset_from_origin": [0, 0],
    # Offset from origin in mm
    "unit": "mm",
    # Unit of the geometry
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
    "feedrate": 1800, "fixed_feedrate": True,
    # Feedrate of the extruder in mm/min. Expected to be an integer
    "pumpspeed": 5,
    "ramp": True
}

params2 = {
    "offset_from_origin": [0, 0, 0],
    # Offset from origin in mm
    "unit": "mm",
    # Tool number of the extruder. Expected to be an integer
    "feedrate": 1800, "fixed_feedrate": True,
    # Feedrate of the extruder in mm/min. Expected to be an integer
    "pumpspeed": 5,
}

@pytest.mark.parametrize("ramp", [False, True])
@pytest.mark.parametrize("standard", ["ConcretePrinter", "ConcretePrinter_BAM"])
def test_gcode(tmp_path, standard:str, ramp:bool):
    caller_path = Path(os.path.dirname(__file__))
    file_point = caller_path / "RandomPoints.csv"
    params1["standard"] = standard
    params1["ramp"] = ramp
    gcd = GcodeFromPoints(**params1)
    file_gcode = Path(tmp_path) / f"test_{standard}.gcode"
    gcd.create(file_point, file_gcode)
    assert file_gcode.exists() & (file_gcode.stat().st_size > 0)

@pytest.mark.parametrize("standard", ["ConcretePrinter", "ConcretePrinter_BAM"])
def test_gcode_3dpoints(tmp_path, standard:str):
    caller_path = Path(os.path.dirname(__file__))
    file_point = caller_path / "RandomPoints3.csv"
    params2["standard"] = standard
    gcd = GcodeFromPoints(**params2)
    file_gcode = Path(tmp_path) / f"test_{standard}.gcode"
    gcd.create(file_point, file_gcode)
    assert file_gcode.exists() & (file_gcode.stat().st_size > 0)

# main
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    test_gcode(Path.cwd(), 'ConcretePrinter',True)
    test_gcode(Path.cwd(), 'ConcretePrinter_BAM', True)

    test_gcode_3dpoints(Path.cwd(), 'ConcretePrinter')
    test_gcode_3dpoints(Path.cwd(), 'ConcretePrinter_BAM')