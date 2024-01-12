from pathlib import Path

import os
import pytest
import h5py
from dolfinx.io import XDMFFile
from fenicsxconcrete.util import ureg
from mpi4py import MPI

from amworkflow.geometry import GeometryParamWall
from amworkflow.meshing import MeshingGmsh
from amworkflow.simulation import SimulationFenicsXConcrete



def test_simulation_structure(tmp_path):

    # tmp output files directory
    d = tmp_path / f"test_sim"
    d.mkdir(parents=True, exist_ok=True)
    file_xdmf_sim = d / f"test_sim.xdmf"

    # use existing meshing file (the one from example Wall dodo_wall.py)
    file_xdmf = Path(os.path.dirname(__file__)) / "test_simple_mesh.xdmf"

    # define required parameters for simulation step
    params_sim_structure = {
        'mesh_unit': 'mm' * ureg(""),
        # which unit is used in mesh file important since fenicsxconcrete converts all in base units!
        'dim': 3 * ureg(""),
        "degree": 2 * ureg(""),
        'q_degree': 2 * ureg(""),
        'bc_setting': 'fixed_y_bottom' * ureg(""),
        'rho': 2400 * ureg("kg/m^3"),
        'g': 9.81 * ureg("m/s^2"),
        'E': 33000 * ureg("MPa"),
        'nu': 0.2 * ureg(""),
        'top_displacement': -20.0 * ureg("mm"),
        'material_type': 'linear' * ureg(""),
        'experiment_type': 'structure' * ureg(""),
    }

    simulation = SimulationFenicsXConcrete(params_sim_structure)
    simulation.run(file_xdmf, file_xdmf_sim)

    # check if output files exist
    assert file_xdmf_sim.exists()

    # check if file has displacement field with nonzero values
    with XDMFFile(MPI.COMM_WORLD, file_xdmf_sim, "r") as xdmf:
        domain = xdmf.read_mesh(name="Grid")
        assert domain.geometry.x[:].max() > 0.0

    with h5py.File(file_xdmf_sim.parent / 'test_sim.h5', 'r') as ifile:
        y = ifile['/Function/displacement/1']
        # print('disp max', y[:].max())
        # print('disp min', y[:].min())
        assert abs(y[:]).max() > 0.0



def test_simulation_process(tmp_path):

    # tmp output files directory
    d = tmp_path / f"test_sim"
    d.mkdir(parents=True, exist_ok=True)
    file_xdmf_sim = d / f"test_sim.xdmf"

    # use existing meshing file (the one from example Wall dodo_wall.py)
    file_xdmf = Path(os.path.dirname(__file__)) / "test_simple_mesh.xdmf"

    # define required parameters for simulation step
    params_sim_process = {
        'mesh_unit': 'mm' * ureg(""),
        # which unit is used in mesh file important since fenicsxconcrete converts all in base units!
        'dim': 3 * ureg(""),
        "degree": 2 * ureg(""),
        'q_degree': 2 * ureg(""),
        'material_type': 'thixo' * ureg(""),
        "rho": 2070 * ureg("kg/m^3"),  # density of fresh concrete
        "nu": 0.3 * ureg(""),  # Poissons Ratio
        "E_0": 0.0779 * ureg("MPa"),  # Youngs Modulus at age=0
        "R_E": 0 * ureg("Pa/s"),  # Reflocculation (first) rate
        "A_E": 0.00002 * ureg("MPa/s"),  # Structuration (second) rate
        "tf_E": 0 * ureg("s"),  # Reflocculation time (switch point)
        "age_0": 0 * ureg("s"),  # start age of concrete
        # layer parameter
        'layer_height': 10 * ureg("mm"),  # to activate layer by layer
        'num_layers': 20 * ureg(""),
        'time_per_layer': 6 * ureg("s"),  # or velocity and layer thickness
        'num_time_steps_per_layer': 2 * ureg(""),
        'experiment_type': 'process' * ureg(""),
    }

    simulation = SimulationFenicsXConcrete(params_sim_process)
    simulation.run(file_xdmf, file_xdmf_sim)

    # check if output files exist
    assert file_xdmf_sim.exists()

    # check if file has displacement field with nonzero values
    with XDMFFile(MPI.COMM_WORLD, file_xdmf_sim, "r") as xdmf:
        domain = xdmf.read_mesh(name="Grid")
        assert domain.geometry.x[:].max() > 0.0

    with h5py.File(file_xdmf_sim.parent / 'test_sim.h5', 'r') as ifile:
        possible_time_tag = params_sim_process['time_per_layer'].magnitude
        y = ifile[f'/Function/displacement/{possible_time_tag}']
        # print('disp max', y[:].max())
        # print('disp min', y[:].min())
        assert abs(y[:]).max() > 0.0
#



#
# ###
# if __name__ == "__main__":
#
#     test_simulation_structure(Path.cwd())
#     test_simulation_process(Path.cwd())
#

