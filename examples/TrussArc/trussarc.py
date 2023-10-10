import pathlib

from doit import create_after, get_var
from doit.task import clean_targets
from doit.tools import config_changed

import pandas as pd
import numpy as np

from amworkflow.geometry import GeometryCenterline

# > doit    # for execution of all task
# > doit s <taskname> # for specific task
# > doit clean # for deleting task output


# use parameter class from fenicsXconcrete ?? TODO
params = {  'name': 'trussarc',
            'out_dir': pathlib.Path(__file__).parent / 'output',  # TODO datastore stuff??
            'csv_points': 'print110823.csv',
            "layer_thickness": 50, # mm
            "number_of_layers": 10,
            "layer_height": 10, # mm

            "mesh_size_factor": 10,
    # ....
}

OUTPUT = params['out_dir']

def task_create_design():
    """create the design

        if a step file was generated externally you can skip this task

        choose a geometry class or create a new one
        - centerline model -> geometryCenterline with given file where the center line point are stored
        - wall model -> geometryWall with parameters for length, height, width, radius, fill-in
        - new geometry -> create a new geometry class from geometry_base class and change the geometry_spawn method accordingly
    """
    pathlib.Path(OUTPUT).mkdir(parents=True, exist_ok=True)

    out_file = OUTPUT / f"{params['name']}.step" # plus stl

    # load centerline points:
    root = pathlib.Path(__file__).parent
    data = pd.read_csv(root / params['csv_points'], sep=',')
    data['z'] = np.zeros(len(data))  # add z coordinate
    # print(data)
    params["points"] = np.array(data[['x', 'y', 'z']])

    geometry = GeometryCenterline(params)

    return {
        "actions": [geometry.create()],
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
#         - meshing via number of layers or layer height possible --> TODO
#     """
#
#     pathlib.Path(OUTPUT).mkdir(parents=True, exist_ok=True)
#
#     in_file = f"{params['name']}.step"
#     out_file = f"{params['name']}.xdmf" # plus vtk
#
#     # new_meshing = Meshing(params)
#
#     return {
#         "file_dep": [in_file],
#         # "actions": [(new_meshing.create(in_file)],
#         "targets": [out_file],
#         "clean": [clean_targets],
#         "uptodate": [config_changed(params)],
#     }

# @create_after(executed="create_meshing")
# def task_simulation():
#     """ perform FE simulation on a given mesh
#
#         Two options:
#         - simulation of the process with element activation
#             * use class amprocess ...
#             * required parameters in params:
#         - simulation of printed structure with linear elastic material
#             * use class amsimulation
#             * required parameters in params:
#
#     to adapt the boundaries/ loading condition create a new class (child of one of the above) and overwrite the method you need to change
#     """
#
#     pathlib.Path(OUTPUT).mkdir(parents=True, exist_ok=True)
#
#     in_file = OUTPUT / f"{params['name']}.stl"
#     out_file = OUTPUT / f"{params['name']}.gcode"
#
#     return {
#         "file_dep": [in_file],
#         # "actions": [(generate_gcode_simple, [in_file, params])],
#         "targets": [out_file],
#         "clean": [clean_targets],
#         "uptodate": [config_changed(params)],
#     }
#
#     pass
#
# @create_after(executed="create_design")
# def task_gcode():
#     """generate gcode from stl file"""
#
#     pass
#
#
#
#
#
#
