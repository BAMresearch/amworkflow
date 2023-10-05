import pathlib

from doit import create_after, get_var
from doit.task import clean_targets
from doit.tools import config_changed

from amworkflow import geometry
from amworkflow import slicer
from amworkflow import simulation
from amworkflow import meshing


# > doit    # for execution of all task
# > doit s <taskname> # for specific task
# > doit clean # for deleting task output


ROOT = pathlib.Path(__file__).parent
SOURCE = ROOT
OUTPUT = ROOT / "out" # TODO datastore stuff??

# use parameter class from fenicsXconcrete ?? TODO
params = {'name': 'template',
    # ....
}

def task_create_design():
    """create the design

        if a step file was generated externally you can skip this task

        how to create a design:

        - ...
    """
    pathlib.Path(OUTPUT).mkdir(parents=True, exist_ok=True)

    out_file = OUTPUT / f"{params['name']}.step" # plus stl

    return {
        # "actions": [...],
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
        - meshing via number of layers or layer height possible --> TODO
    """

    pathlib.Path(OUTPUT).mkdir(parents=True, exist_ok=True)

    in_file = f"{params['name']}.step"
    out_file = f"{params['name']}.xdmf" # plus vtk

    return {
        "file_dep": [in_file],
        # "actions": [(generate_gcode_simple, [in_file, params])],
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

    pathlib.Path(OUTPUT).mkdir(parents=True, exist_ok=True)

    in_file = OUTPUT / f"{params['name']}.stl"
    out_file = OUTPUT / f"{params['name']}.gcode"

    return {
        "file_dep": [in_file],
        # "actions": [(generate_gcode_simple, [in_file, params])],
        "targets": [out_file],
        "clean": [clean_targets],
        "uptodate": [config_changed(params)],
    }

    pass

@create_after(executed="create_design")
def task_gcode():
    """generate gcode from stl file"""

    pass






