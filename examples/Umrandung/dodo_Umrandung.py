import datetime
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from doit import create_after, get_var
from doit.task import clean_targets
from doit.tools import config_changed

from fenicsxconcrete.util import ureg

from amworkflow.geometry import GeometryCenterline
from amworkflow.meshing import MeshingGmsh
from amworkflow.gcode import GcodeFromPoints
from amworkflow.simulation import SimulationFenicsXConcrete

# > doit -f <filename>   # for execution of all task
# > doit -f <filename> s <taskname> # for specific task
# > doit -f <filename> clean # for deleting task output

logging.basicConfig(level=logging.INFO)

params_gcode = {  # gcode parameters
    "layer_width":  10,
    "unit": "mm",    # Unit of the geometry
    "standard": "ConcretePrinter_BAM",   # Standard of the printer firmware TU printer
    "coordinate_system": "absolute", # Coordinate system of the printer firmware
    "nozzle_diameter": 10, # Diameter of the nozzle in mm
    "feedrate": 10800,
    "fixed_feedrate": True,
    "pumpspeed": 0
}

# TODO datastore stuff??
OUTPUT_NAME = Path(__file__).parent.name
OUTPUT = (
    Path(__file__).parent / "output"
)  # / f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'


def task_gcode():
    """Generate machine code (gcode) for design from a point csv file."""

    OUTPUT.mkdir(parents=True, exist_ok=True)

    in_file_points = Path(__file__).parent  / f"{OUTPUT_NAME}.csv"
    out_file_gcode = OUTPUT / f"{OUTPUT_NAME}.nc"

    gcd = GcodeFromPoints(**params_gcode)

    return {
        "file_dep": [in_file_points],
        "actions": [(gcd.create, [in_file_points, out_file_gcode])],
        "targets": [out_file_gcode],
        "clean": [clean_targets],
        "uptodate": [config_changed(params_gcode)],
        "verbosity": 1,
    }

if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
