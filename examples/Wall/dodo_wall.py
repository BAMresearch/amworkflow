from pathlib import Path

from doit import create_after, get_var
from doit.task import clean_targets
from doit.tools import config_changed

import pandas as pd
import numpy as np

from amworkflow.geometry import GeometryParamWall
from amworkflow.meshing import Meshing

# > doit -f <filename>   # for execution of all task
# > doit -f <filename> s <taskname> # for specific task
# > doit -f <filename> clean # for deleting task output


# use parameter class from fenicsXconcrete ?? TODO
params =   {'name': 'wall',
            'out_dir': str(Path(__file__).parent / 'output'),  # TODO datastore stuff??
            "length": 200, # mm
            "height": 200, # mm
            "width": 200, # mm
            "radius": 1000, # mm
            "infill": "solid",
            "mesh_size_factor": 10}

OUTPUT = Path(params['out_dir'])

def task_create_design():
    """create the design

        if a step file was generated externally you can skip this task

        choose a geometry class or create a new one
        - centerline model -> geometryCenterline with given file where the center line point are stored
        - wall model -> geometryWall with parameters for length, height, width, radius, fill-in
        - new geometry -> create a new geometry class from geometry_base class and change the geometry_spawn method accordingly
    """
    OUTPUT.mkdir(parents=True, exist_ok=True)

    out_file = OUTPUT / f"{params['name']}.stp" # plus stl, plus path points!!!!
    print(params)

    geometry = GeometryParamWall(params)

    return {
        "actions": [(geometry.create, [])],
        "targets": [out_file],
        "clean": [clean_targets],
        "uptodate": [config_changed(params)],
    }


# @create_after(executed="create_design")
# def task_meshing():
#     """meshing a given design from a step file
#
#         required parameters in params:
#         - name: name of the design
#         - mesh_size: size of the mesh
#         - meshing via number of layers or layer height possible
#     """
#
#     OUTPUT.mkdir(parents=True, exist_ok=True)
#
#     in_file = f"{params['name']}.step"
#     out_file = f"{params['name']}.xdmf" # plus vtk
#
#     new_meshing = Meshing(in_file, params)
#
#     return {
#         "file_dep": [in_file],
#         "actions": [(new_meshing.create, [in_file])],
#         "targets": [out_file],
#         "clean": [clean_targets],
#         "uptodate": [config_changed(params)],
#     }
