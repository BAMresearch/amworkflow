from pathlib import Path

import pytest

from amworkflow.geometry import GeometryParamWall
from amworkflow.meshing import MeshingGmsh


def test_meshingGmsh(tmp_path):
    # define required parameters for simple design
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

    # output files in tmp directory
    d = tmp_path / "test_paramwall_solid"
    d.mkdir(parents=True, exist_ok=True)
    file_step = d / f"test.stp"
    file_points = d / f"test.csv"
    file_xdmf = d / f"test.xdmf"
    file_vtk = d / f"test.vtk"

    # need geometry step file to create mesh
    geometry = GeometryParamWall(**params)
    geometry.create(file_step, file_points)

    meshing = MeshingGmsh(**params)
    meshing.create(file_step, file_xdmf, file_vtk)

    # check if output files exist
    assert file_xdmf.exists()
    assert file_vtk.exists()

###
# if __name__ == "__main__":
#     test_meshingGmsh(Path.cwd())