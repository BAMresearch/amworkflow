from pathlib import Path
import datetime
import logging

from doit import create_after, get_var
from doit.task import clean_targets
from doit.tools import config_changed

import pandas as pd
import numpy as np

from amworkflow.geometry import GeometryParamWall
from amworkflow.meshing import MeshingGmsh

# > doit -f <filename>   # for execution of all task
# > doit -f <filename> s <taskname> # for specific task
# > doit -f <filename> clean # for deleting task output

logging.basicConfig(level=logging.INFO)

# define required parameters
params = {  # geometry parameters
    "length": 200,  # mm
    "height": 200,  # mm
    "width": 200,  # mm
    "radius": 1000,  # mm
    "infill": "solid",
    # mesh parameters (meshing by layer height)
    "mesh_size_factor": 10,
    "layer_height": 10,  # mm
}

# TODO datastore stuff??
OUTPUT_NAME = Path(__file__).parent.name
OUTPUT = (
    Path(__file__).parent / "output"
)  # / f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'


def task_create_design():
    """Create the design of the parametric wall."""

    OUTPUT.mkdir(parents=True, exist_ok=True)

    out_file_step = OUTPUT / f"{OUTPUT_NAME}.stp"
    out_file_stl = OUTPUT / f"{OUTPUT_NAME}.stl"
    out_file_points = OUTPUT / f"{OUTPUT_NAME}.csv"

    geometry = GeometryParamWall(**params)

    return {
        "actions": [(geometry.create, [out_file_step, out_file_points, out_file_stl])],
        "targets": [out_file_step, out_file_points, out_file_stl],
        "clean": [clean_targets],
        "uptodate": [config_changed(params)],
    }


@create_after(executed="create_design")
def task_meshing():
    """Meshing a given design from a step file."""

    OUTPUT.mkdir(parents=True, exist_ok=True)

    in_file_step = OUTPUT / f"{OUTPUT_NAME}.stp"
    out_file_xdmf = OUTPUT / f"{OUTPUT_NAME}.xdmf"
    out_file_vtk = OUTPUT / f"{OUTPUT_NAME}.vtk"

    meshing = MeshingGmsh(**params)

    return {
        "file_dep": [in_file_step],
        "actions": [(meshing.create, [in_file_step, out_file_xdmf, out_file_vtk])],
        "targets": [out_file_xdmf, out_file_vtk],
        "clean": [clean_targets],
        "uptodate": [config_changed(params)],
    }
