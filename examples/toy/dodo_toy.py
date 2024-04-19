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

# define required parameters
params = {  # geometry parameters
    "layer_thickness": 10,  # mm
    "height": 40,  # mm
    # mesh parameters (meshing by layer height)
    "mesh_size_factor": 1,
    # "layer_height": 10,  # mm
    "number_of_layers": 4,
}
params_gcode = {  # gcode parameters
    "layer_num": 4,
    "layer_height": 10, #mm
    "layer_width":  params["layer_thickness"],
    "offset_from_origin": [0, 0], # Offset from origin in mm
    "unit": "mm",    # Unit of the geometry
    "standard": "ConcretePrinter",   # Standard of the printer firmware
    "coordinate_system": "absolute", # Coordinate system of the printer firmware
    "nozzle_diameter": 0.4, # Diameter of the nozzle in mm
    "kappa": 1, # Parameter for the calculation of the extrusion width
    "tool_number": 0, # Tool number of the extruder. Expected to be an integer
    "feedrate": 1800,
}

# simulation parameters needs to be in pint units!!
params_sim_structure = {
    "mesh_unit": "mm"
    * ureg(
        ""
    ),  # which unit is used in mesh file important since fenicsxconcrete converts all in base units!
    "dim": 3 * ureg(""),
    "degree": 2 * ureg(""),
    "q_degree": 2 * ureg(""),
    "bc_setting": "fixed_y" * ureg(""),
    "rho": 2400 * ureg("kg/m^3"),
    "g": 9.81 * ureg("m/s^2"),
    "E": 33000 * ureg("MPa"),
    "nu": 0.2 * ureg(""),
    "top_displacement": -5.0 * ureg("mm"),
    "number_steps": 1 * ureg(""),
    "material_type": "linear" * ureg(""),
}

# TODO datastore stuff??
OUTPUT_NAME = Path(__file__).parent.name
OUTPUT = (
    Path(__file__).parent / "output"
)  # / f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'


def task_create_design():
    """create the design of a centerline model"""

    OUTPUT.mkdir(parents=True, exist_ok=True)

    out_file_step = OUTPUT / f"{OUTPUT_NAME}.stp"
    out_file_stl = OUTPUT / f"{OUTPUT_NAME}.stl"
    out_file_points = OUTPUT / f"{OUTPUT_NAME}.csv"

    # define center line points here for this example:
    points = [[0., 0., 0.],
              [0., 150., 0.],
              [10., 150., 0],
              [75., 75., 0.],
              [140., 150., 0.],
              [150., 150., 0.],
              [150., 0., 0.]]
    params["points"] = points
    geometry = GeometryCenterline(**params)

    return {
        "actions": [(geometry.create, [out_file_step,  out_file_points, out_file_stl])],
        "targets": [out_file_step, out_file_stl, out_file_points],
        "clean": [clean_targets],
        "uptodate": [config_changed(params)],
    }

@create_after(executed="create_design")
def task_gcode():
    """Generate machine code (gcode) for design from a point csv file."""

    OUTPUT.mkdir(parents=True, exist_ok=True)

    in_file_points = OUTPUT / f"{OUTPUT_NAME}.csv"
    out_file_gcode = OUTPUT / f"{OUTPUT_NAME}.gcode"

    gcd = GcodeFromPoints(**params_gcode)

    return {
        "file_dep": [in_file_points],
        "actions": [(gcd.create, [in_file_points, out_file_gcode])],
        "targets": [out_file_gcode],
        "clean": [clean_targets],
        "uptodate": [config_changed(params_gcode)],
    }

# @create_after(executed="create_design")
# def task_powderbed_code():
#     """Generate print instructions for BAM powder bed printer"""

#     return {
#         "file_dep": [in_file_points],
#         "actions": [(gcd.create, [in_file_points, out_file_gcode])],
#         "targets": [out_file_gcode],
#         "clean": [clean_targets],
#         "uptodate": [config_changed(params)],
#     }

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

@create_after(executed="meshing")
def task_structure_simulation():
    """Simulating the final structure loaded parallel to layers in y-direction with displacement."""

    OUTPUT.mkdir(parents=True, exist_ok=True)

    in_file_xdmf = OUTPUT / f"{OUTPUT_NAME}.xdmf"
    out_file_xdmf = OUTPUT / f"{OUTPUT_NAME}_sim_structure.xdmf"

    params_sim_structure["experiment_type"] = "structure" * ureg("")
    simulation = SimulationFenicsXConcrete(params_sim_structure)

    return {
        "file_dep": [in_file_xdmf],
        "actions": [(simulation.run, [in_file_xdmf, out_file_xdmf])],
        "targets": [out_file_xdmf],
        "clean": [clean_targets],
        # "uptodate": [config_changed(params_sim)], # param_sim not possible for doit
        "verbosity": 2,
    }

if __name__ == "__main__":

    OUTPUT.mkdir(parents=True, exist_ok=True)

    out_file_step = OUTPUT / f"{OUTPUT_NAME}.stp"
    out_file_stl = OUTPUT / f"{OUTPUT_NAME}.stl"
    out_file_points = OUTPUT / f"{OUTPUT_NAME}.csv"

    # define center line points here for this example:
    points = [[0., 0., 0.],
              [0., 150., 0.],
              [10., 150., 0],
              [75., 75., 0.],
              [140., 150., 0.],
              [150., 150., 0.],
              [150., 0., 0.]]
    params["points"] = points
    geometry = GeometryCenterline(**params)

    in_file_points = OUTPUT / f"{OUTPUT_NAME}.csv"
    out_file_gcode = OUTPUT / f"{OUTPUT_NAME}.gcode"

    gcd = GcodeFromPoints(**params_gcode, in_file_path=in_file_points)

    AAA = GcodeFromPoints.create(gcd, in_file=in_file_points, out_gcode=out_file_gcode)
    BBB = 1