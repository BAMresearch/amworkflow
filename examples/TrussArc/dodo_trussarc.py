import datetime
import logging
from pathlib import Path

from fenicsxconcrete.util import ureg
import numpy as np
import pandas as pd
from doit import create_after, get_var
from doit.task import clean_targets
from doit.tools import config_changed

from amworkflow.geometry import GeometryCenterline
from amworkflow.meshing import MeshingGmsh
from amworkflow.simulation import SimulationFenicsXConcrete

# > doit -f <filename>   # for execution of all task
# > doit -f <filename> s <taskname> # for specific task
# > doit -f <filename> clean # for deleting task output

logging.basicConfig(level=logging.WARNING)

# define required parameters
params = {  # geometry parameters
    "csv_points": "print110823.csv",
    "layer_thickness": 50,  # mm
    "height": 100,  # mm
    # mesh parameters (meshing by layer height)
    "mesh_size_factor": 10,
    "number_of_layers": 10,
    "is_close": False,
}

#parameters for the simulation of process
params_sim_process = {
    "mesh_unit": "mm"
    * ureg(
        ""
    ),  # which unit is used in mesh file important since fenicsxconcrete converts all in base units!
    "dim": 3 * ureg(""), # dimension of the simulation
    "degree": 2 * ureg(""), # degree of the finite element
    "q_degree": 2 * ureg(""), # degree of the quadrature
    #
    "material_type": "thixo" * ureg(""),  # material type
    "experiment_type": "process" * ureg(""),  # type of the experiment
    #
    "g": 9.81 * ureg("m/s^2"), # gravity
    "rho": 2070 * ureg("kg/m^3"), # density of the material -> no body force in the moment!
    "nu": 0.3 * ureg(""), # Poisson's ratio
    "E_0": 0.0779 * ureg("MPa"),  # Youngs Modulus at age=0
    "R_E": 0 * ureg("Pa/s"),  # Reflocculation (first) rate
    "A_E": 0.00002 * ureg("MPa/s"),  # Structuration (second) rate
    "tf_E": 0 * ureg("s"),  # Reflocculation time (switch point)
    "age_0": 0 * ureg("s"),  # start age of concrete
    # layer parameter
    'layer_height': 10 * ureg("mm"),  # to activate layer by layer
    'num_layers': 10 * ureg(""),
    'layer_thickness': 50 * ureg("mm"),  # for computing path length and with velocity time for one layer
    'print_velocity': 138 * ureg("mm/s"), # to get approx 60s /Layer
    'num_time_steps_per_layer': 2 * ureg(""),

}

params_sim_structure = {
    "mesh_unit": "mm"
    * ureg(
        ""
    ),  # which unit is used in mesh file important since fenicsxconcrete converts all in base units!
    "dim": 3 * ureg(""), # dimension of the simulation
    "degree": 2 * ureg(""), # degree of the finite element
    "q_degree": 2 * ureg(""), # degree of the quadrature
    #
    "material_type": "linear" * ureg(""),  # material type
    "experiment_type": "structure" * ureg(""),  # type of the experiment

    'bc_setting': 'fixed_truss' * ureg(""),
    'rho': 2400 * ureg("kg/m^3"),
    'g': 9.81 * ureg("m/s^2"),
    'E': 33000 * ureg("MPa"),
    'nu': 0.2 * ureg(""),
    'top_displacement': 0.0 * ureg("mm"), # max displacement of top surface
     "number_steps": 1 * ureg(""), # number of steps for simulation

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

    # load centerline points:
    data = pd.read_csv(Path(__file__).parent / params["csv_points"], sep=",")
    data["z"] = np.zeros(len(data))  # add z coordinate
    # print(data)
    # params["points"] = np.array(data[['x', 'y', 'z']])
    point_list = data[["x", "y", "z"]].values.tolist()
    params["points"] = point_list
    geometry = GeometryCenterline(**params)

    return {
        "actions": [(geometry.create, [out_file_step, out_file_points, out_file_stl])],
        "targets": [out_file_step, out_file_stl, out_file_points],
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

@create_after(executed="meshing")
def task_process_simulation():
    """Simulating of the process."""

    OUTPUT.mkdir(parents=True, exist_ok=True)

    in_file_xdmf = OUTPUT / f"{OUTPUT_NAME}.xdmf"
    out_file_xdmf = OUTPUT / f"{OUTPUT_NAME}_sim_process.xdmf"

    # displacment load in y direction
    simulation = SimulationFenicsXConcrete(params_sim_process)

    return {
        "file_dep": [in_file_xdmf],
        "actions": [(simulation.run, [in_file_xdmf, out_file_xdmf])],
        "targets": [out_file_xdmf],
        "clean": [clean_targets],
        "verbosity": 2,
    }

# @create_after(executed="meshing")
# def task_structure_simulation():
#     """Simulating the final structure under dead load."""
#
#     OUTPUT.mkdir(parents=True, exist_ok=True)
#
#     in_file_xdmf = OUTPUT / f"{OUTPUT_NAME}.xdmf"
#     out_file_xdmf = OUTPUT / f"{OUTPUT_NAME}_sim_structure.xdmf"
#
#     simulation = SimulationFenicsXConcrete(params_sim_structure)
#
#     return {
#         "file_dep": [in_file_xdmf],
#         "actions": [(simulation.run, [in_file_xdmf, out_file_xdmf])],
#         "targets": [out_file_xdmf],
#         "clean": [clean_targets],
#         # "uptodate": [config_changed(params_sim)], # param_sim not possible for doit
#         "verbosity": 2,
#     }

if __name__ == "__main__":
    import doit

    doit.run(globals())
