import datetime
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from doit import create_after, get_var
from doit.task import clean_targets
from doit.tools import config_changed

from amworkflow.geometry import GeometryParamWall
from amworkflow.meshing import MeshingGmsh

# from fenicsxconcrete.util import ureg


# from amworkflow.simulation import SimulationFenicsXConcrete

# > doit -f <filename>   # for execution of all task
# > doit -f <filename> s <taskname> # for specific task
# > doit -f <filename> clean # for deleting task output
# add infill=<zigzag/solid/honeycomb> default is zigzag

logging.basicConfig(level=logging.INFO)

# define required parameters
params = {  # geometry parameters
    "length": 150,  # mm
    "height": 150,  # mm
    "width": 150,  # mm
    "radius": None,  # mm
    "infill": get_var('infill',"zigzag"), # default infill changable via command line doit -f dodo_wall.py infill=zigzag or solid or honeycomb
    # mesh parameters (meshing by layer height)
    "line_width": 10,  # mm
    "mesh_size_factor": 10,
    "layer_height": 10,  # mm
}
# # simulation parameters needs to be in pint units!!
# params_sim_structure = {
#     "mesh_unit": "mm"
#     * ureg(
#         ""
#     ),  # which unit is used in mesh file important since fenicsxconcrete converts all in base units!
#     "dim": 3 * ureg(""),
#     "degree": 2 * ureg(""),
#     "q_degree": 2 * ureg(""),
#     "bc_setting": "fixed_y_bottom" * ureg(""),
#     "rho": 2400 * ureg("kg/m^3"),
#     "g": 9.81 * ureg("m/s^2"),
#     "E": 33000 * ureg("MPa"),
#     "nu": 0.2 * ureg(""),
#     "top_displacement": -20.0 * ureg("mm"),
#     "material_type": "linear" * ureg(""),
# }
# params_sim_process = {
#     "mesh_unit": "mm"
#     * ureg(
#         ""
#     ),  # which unit is used in mesh file important since fenicsxconcrete converts all in base units!
#     "dim": 3 * ureg(""),
#     "degree": 2 * ureg(""),
#     "q_degree": 2 * ureg(""),
#     "material_type": "thixo" * ureg(""),
#     "rho": 2070 * ureg("kg/m^3"),  # density of fresh concrete
#     "nu": 0.3 * ureg(""),  # Poissons Ratio
#     "E_0": 0.0779 * ureg("MPa"),  # Youngs Modulus at age=0
#     "R_E": 0 * ureg("Pa/s"),  # Reflocculation (first) rate
#     "A_E": 0.00002 * ureg("MPa/s"),  # Structuration (second) rate
#     "tf_E": 0 * ureg("s"),  # Reflocculation time (switch point)
#     "age_0": 0 * ureg("s"),  # start age of concrete
#     # layer parameter
#     "layer_height": params["layer_height"] * ureg("mm"),  # to activate layer by layer
#     "num_layers": params["height"] / params["layer_height"] * ureg(""),
#     "time_per_layer": 6 * ureg("s"),  # or velocity and layer thickness
#     "num_time_steps_per_layer": 2 * ureg(""),
# }

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
        "verbosity": 2,
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
        "verbosity": 2,
    }


# @create_after(executed="meshing")
# def task_structure_simulation():
#     """Simulating the final structure loaded parallel to layers in y-direction with displacement."""

#     OUTPUT.mkdir(parents=True, exist_ok=True)

#     in_file_xdmf = OUTPUT / f"{OUTPUT_NAME}.xdmf"
#     out_file_xdmf = OUTPUT / f"{OUTPUT_NAME}_sim_structure.xdmf"

#     params_sim_structure["experiment_type"] = "structure" * ureg("")
#     simulation = SimulationFenicsXConcrete(params_sim_structure)

#     return {
#         "file_dep": [in_file_xdmf],
#         "actions": [(simulation.run, [in_file_xdmf, out_file_xdmf])],
#         "targets": [out_file_xdmf],
#         "clean": [clean_targets],
#         # "uptodate": [config_changed(params_sim)], # param_sim not possible for doit
#         "verbosity": 2,
#     }


# @create_after(executed="meshing")
# def task_process_simulation():
#     """Simulating the final structure loaded parallel to layers."""

#     OUTPUT.mkdir(parents=True, exist_ok=True)

#     in_file_xdmf = OUTPUT / f"{OUTPUT_NAME}.xdmf"
#     out_file_xdmf = OUTPUT / f"{OUTPUT_NAME}_sim_process.xdmf"

#     params_sim_process["experiment_type"] = "process" * ureg("")
#     simulation = SimulationFenicsXConcrete(params_sim_process)

#     return {
#         "file_dep": [in_file_xdmf],
#         "actions": [(simulation.run, [in_file_xdmf, out_file_xdmf])],
#         "targets": [out_file_xdmf],
#         "clean": [clean_targets],
#         # "uptodate": [config_changed(params_sim)], # param_sim not possible for doit
#         "verbosity": 2,
#     }
