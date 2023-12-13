import logging
from pathlib import Path

from doit import get_var
from doit.task import clean_targets
from doit.tools import config_changed

import amworkflow.gcode.gcode as gcode

# > doit -f <filename>   # for execution of all task
# > doit -f <filename> s <taskname> # for specific task
# > doit -f <filename> clean # for deleting task output

logging.basicConfig(level=logging.INFO)

# define required parameters
params = {  # geometry parameters
    "layer_num": 1,
    # Number of printed layers. expected to be an integer
    "layer_height": 1,
    # Layer height in mm
    "line_width": 1,
    # Line width in mm
    "offset_from_origin": None,
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
    "in_file_path": "",
    # Path to the input file
}

# TODO datastore stuff??
OUTPUT_NAME = Path(__file__).parent.name
OUTPUT = (
    Path(__file__).parent / "output"
)  # / f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'


def task_create_gcode():
    """Create the design of the parametric wall."""

    OUTPUT.mkdir(parents=True, exist_ok=True)

    out_file_gcode = OUTPUT / f"{OUTPUT_NAME}.gcode"

    gcd = gcode.GcodeFromPoints(**params)

    return {
        "actions": [(gcd.create, [params["in_file_path"], out_file_gcode])],
        "targets": [out_file_gcode],
        "clean": [clean_targets],
        "uptodate": [config_changed(params)],
        "verbosity": 2,
    }
