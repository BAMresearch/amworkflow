from pathlib import Path

from doit import create_after, get_var
from doit.task import clean_targets
from doit.tools import config_changed

# from amworkflow.geometry import ...

# > doit    # for execution of all task
# > doit s <taskname> # for specific task
# > doit clean # for deleting task output

# use parameter class from fenicsXconcrete ?? TODO
params = {'name': 'template',
          'out_dir': str(Path(__file__).parent / 'output'),  # TODO datastore stuff??
    # ....
}

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

    out_file = OUTPUT / f"{params['name']}.step" # plus stl

    # geometry = ...(params)

    return {
        "actions": [(geometry.create,[])],
        "targets": [out_file],
        "clean": [clean_targets],
        "uptodate": [config_changed(params)],
    }


@create_after(executed="create_design")
def task_meshing():
    """meshing a given design from a step file

        required parameters in params:
        - name: name of the design
        - mesh_size: size of the mesh
        - meshing via number of layers or layer height possible
    """

    OUTPUT.mkdir(parents=True, exist_ok=True)

    in_file = f"{params['name']}.step"
    out_file = f"{params['name']}.xdmf" # plus vtk

    # new_meshing = ...(params)

    return {
        "file_dep": [in_file],
        "actions": [(new_meshing.create, [in_file])],
        "targets": [out_file],
        "clean": [clean_targets],
        "uptodate": [config_changed(params)],
    }

@create_after(executed="create_meshing")
def task_simulation():
    """ perform FE simulation on a given mesh

        Two options:
        - simulation of the process with element activation
            * use class amprocess ...
            * required parameters in params:
        - simulation of printed structure with linear elastic material
            * use class amsimulation
            * required parameters in params:

    to adapt the boundaries/ loading condition create a new class (child of one of the above) and overwrite the method you need to change
    """

    OUTPUT.mkdir(parents=True, exist_ok=True)

    in_file = OUTPUT / f"{params['name']}.stl"
    out_file = OUTPUT / f"{params['name']}.gcode"

    # gcode = ...(params)

    return {
        "file_dep": [in_file],
        "actions": [(gcode.create, [in_file])],
        "targets": [out_file],
        "clean": [clean_targets],
        "uptodate": [config_changed(params)],
    }

    pass

@create_after(executed="create_design")
def task_gcode():
    """generate gcode from stl file"""

    pass






