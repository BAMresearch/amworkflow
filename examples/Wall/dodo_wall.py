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
from amworkflow.gcode import GcodeFromPoints
from amworkflow.simulation import SimulationFenicsXConcrete
from fenicsxconcrete.util import ureg

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
    "infill": get_var(
        "infill", "honeycomb"
    ),  # default infill changeable via command line doit -f dodo_wall.py infill=zigzag or solid or honeycomb
    # mesh parameters (meshing by layer height)
    "line_width": float(
        get_var("line_width", 10)
    ),  # mm # 11 for zigzag 10 for honeycomb to get same volume reduction
    "mesh_size_factor": float(
        get_var("mesh_size", 4)
    ),  # default mesh size factor changeable via command line doit -f dodo_wall.py mesh_size=4
    "layer_height": 10,  # mm
}
params_gcode = {  # gcode parameters
    "layer_num": int(params["height"]/params["layer_height"]),
    "layer_height": params["layer_height"], #mm
    "layer_width":  params["line_width"],
    "offset_from_origin": [0, 0], # Offset from origin in mm
    "unit": "mm",    # Unit of geometry
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
    "dim": 3 * ureg(""), # dimension of the simulation
    "degree": 2 * ureg(""), # degree of the finite element
    "q_degree": 2 * ureg(""), # degree of the quadrature
    "g": 9.81 * ureg("m/s^2"), # gravity
    "rho": 0 * ureg("kg/m^3"), # density of the material -> no body force in the moment!
    "E": 30000 * ureg("MPa"), # Young's modulus (usually concrete: 33000)
    "nu": 0.2 * ureg(""), # Poisson's ratio
    #"bc_setting": "compr_disp_y" * ureg(""), # bc setting for structure simulation -> defined in task_structure_simulation_...
    "top_displacement": -1.5 * ureg("mm"), # max displacement of top surface
    "number_steps": 3 * ureg(""), # number of steps for simulation
    "material_type": "linear" * ureg(""), # material type
    "experiment_type": "structure" * ureg(""), # type of the experiment
}

# TODO datastore stuff??
OUTPUT_NAME = (
    f"{Path(__file__).parent.name}_{params['infill']}_mesh_{params['mesh_size_factor']}"
)
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


@create_after(executed="meshing")
def task_structure_simulation_disp_y():
    """Simulating the final structure loaded parallel to layers in y-direction with displacement."""

    OUTPUT.mkdir(parents=True, exist_ok=True)

    in_file_xdmf = OUTPUT / f"{OUTPUT_NAME}.xdmf"
    out_file_xdmf = OUTPUT / f"{OUTPUT_NAME}_sim_disp_y.xdmf"

    # displacment load in y direction
    params_sim_structure["bc_setting"] = "compr_disp_y" * ureg("")
    simulation = SimulationFenicsXConcrete(params_sim_structure)

    return {
        "file_dep": [in_file_xdmf],
        "actions": [(simulation.run, [in_file_xdmf, out_file_xdmf])],
        "targets": [out_file_xdmf],
        "clean": [clean_targets],
        # "uptodate": [config_changed(params_sim)], # param_sim not possible for doit
        "verbosity": 2,
    }


@create_after(executed="meshing")
def task_structure_simulation_disp_x():
    """Simulating the final structure loaded parallel to layers in x-direction with displacement."""

    OUTPUT.mkdir(parents=True, exist_ok=True)

    in_file_xdmf = OUTPUT / f"{OUTPUT_NAME}.xdmf"
    out_file_xdmf = OUTPUT / f"{OUTPUT_NAME}_sim_disp_x.xdmf"

    # displacment load in y direction
    params_sim_structure["bc_setting"] = "compr_disp_x" * ureg("")
    simulation = SimulationFenicsXConcrete(params_sim_structure)

    return {
        "file_dep": [in_file_xdmf],
        "actions": [(simulation.run, [in_file_xdmf, out_file_xdmf])],
        "targets": [out_file_xdmf],
        "clean": [clean_targets],
        # "uptodate": [config_changed(params_sim)], # param_sim not possible for doit
        "verbosity": 2,
    }
