import pytest
from pathlib import Path


from amworkflow.geometry import GeometryParamWall

def test_geometry_paramwall_solid(tmp_path):
    # define required geometry parameters for simple design
    params = {
                "length": 200, # mm
                "height": 200, # mm
                "width": 200, # mm
                "radius": 1000, # mm
                "infill": "solid",
    }

    # output files in tmp directory
    d = tmp_path / "test_paramwall_solid"
    d.mkdir(parents=True, exist_ok=True)
    file_step = d / f"test.stp"
    file_stl = d / f"test.stl"
    file_points = d / f"test.csv"

    geometry = GeometryParamWall(**params)
    geometry.create(file_step, file_points, file_stl)

    # check if output files exist
    assert file_step.exists()
    assert file_stl.exists()
    # assert out_file_points.exists() #not yet implemented

###
# if __name__ == "__main__":
#     test_geometry_paramwall_solid(Path.cwd())