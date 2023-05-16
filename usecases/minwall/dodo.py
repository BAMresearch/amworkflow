import pathlib

from convert_mesh import convert_msh2xdmf
from doit import create_after, get_var
from doit.action import CmdAction
from doit.task import clean_targets
from doit.tools import config_changed
from gcode import generate_gcode_simple
from simulation import amsimulation

# > doit    # for execution of all task
# > doit s <taskname> # for specific task
# > doit clean # for deleting task output
# display workflow using "doit-graph":
# > doit graph
# > dot -Tpng tasks.dot -o task.png

ROOT = pathlib.Path(__file__).parent
SOURCE = ROOT
OUTPUT = ROOT / "out"

# TODO: read from general yaml for each example?
GLOBAL_PARAMS = {
    "file_name": "wall_zigzag",
    # "file_name": "wall",
    ### mesh parameters
    # geometry dimensions of wall structure in length unit e.g. mm!
    "length": 2 * 60.0,  # from terminal by get_var("length", "2.0")
    "width": 2 * 6.0,
    "height": 8,
    "layer_width": 2 * 1.0,
    "layer_height": 1.0,  # multiple from height!!
    # mesh density
    "meshSize": 1,
    ### slicer parameters
    "velocity": 36,  # mm/s
    "feed": 60,  # mm extrusion/mm path
    "machine zero point": [50, 100],  # (x,y)
    ### simulation parameters
    "dt": 1.0,  # s
}

DOIT_CONFIG = {
    "action_string_formatting": "both",
    "verbosity": 2,
}

#TODO: genarlize those task independent of given parameter sets via json?
def task_generate_stl_mesh():
    """generate the surface mesh with Gmsh"""
    pathlib.Path(OUTPUT).mkdir(parents=True, exist_ok=True)

    geo = SOURCE / f"{GLOBAL_PARAMS['file_name']}.geo"
    stl = OUTPUT / f"{GLOBAL_PARAMS['file_name']}.stl"
    args = [
        "gmsh",
        "-2",  # 2D meshing!
        "-setnumber",
        "stl",  # for stl variante
        "1",
        "-setnumber",
        "length",
        f"{GLOBAL_PARAMS['length']}",
        "-setnumber",
        "width",
        f"{GLOBAL_PARAMS['width']}",
        "-setnumber",
        "height",
        f"{GLOBAL_PARAMS['height']}",
        "-setnumber",
        "layer_width",
        f"{GLOBAL_PARAMS['layer_width']}",
        "-setnumber",
        "layer_height",
        f"{GLOBAL_PARAMS['layer_height']}",
        "-setnumber",
        "meshSize",
        f"{GLOBAL_PARAMS['meshSize']}",
        f"{geo}",
        "-o",
        f"{stl}",
    ]
    return {
        "file_dep": [geo],
        # "actions": [CmdAction(" ".join(args), save_out="stdout")],
        "actions": [CmdAction(" ".join(args))],
        "targets": [stl],
        "clean": [clean_targets],
        "uptodate": [config_changed(GLOBAL_PARAMS)],
    }


@create_after(executed="generate_stl_mesh")
def task_generate_gcode():
    """generate gcode from stl file

    should by slicer software like ULTIMKAER CURA/ Creality from stl file via Command line
    here as test the gcode is self-written
    """

    pathlib.Path(OUTPUT).mkdir(parents=True, exist_ok=True)

    stl = OUTPUT / f"{GLOBAL_PARAMS['file_name']}.stl"
    gcode = OUTPUT / f"{GLOBAL_PARAMS['file_name']}.gcode"

    return {
        "file_dep": [stl],
        "actions": [(generate_gcode_simple, [stl, GLOBAL_PARAMS])],
        "targets": [gcode],
        "clean": [clean_targets],
        "uptodate": [config_changed(GLOBAL_PARAMS)],
    }


def task_generate_msh_mesh():
    """generate the surface mesh with Gmsh"""
    pathlib.Path(OUTPUT).mkdir(parents=True, exist_ok=True)

    geo = SOURCE / f"{GLOBAL_PARAMS['file_name']}.geo"
    msh = OUTPUT / f"{GLOBAL_PARAMS['file_name']}.msh"
    args = [
        "gmsh",
        "-3",  # 3D meshing! no set no stl value !!
        "-setnumber",
        "stl",
        "0",  # for msh variante
        "-setnumber",
        "length",
        f"{GLOBAL_PARAMS['length']}",
        "-setnumber",
        "width",
        f"{GLOBAL_PARAMS['width']}",
        "-setnumber",
        "height",
        f"{GLOBAL_PARAMS['height']}",
        "-setnumber",
        "layer_width",
        f"{GLOBAL_PARAMS['layer_width']}",
        "-setnumber",
        "layer_height",
        f"{GLOBAL_PARAMS['layer_height']}",
        "-setnumber",
        "meshSize",
        f"{GLOBAL_PARAMS['meshSize']}",
        f"{geo}",
        "-o",
        f"{msh}",
    ]
    return {
        "file_dep": [geo],
        "actions": [CmdAction(" ".join(args), save_out="stdout")],
        "targets": [msh],
        "clean": [clean_targets],
        "uptodate": [config_changed(GLOBAL_PARAMS)],
    }


@create_after(executed="generate_msh_mesh")
def task_convert_msh_xdmf():
    """convert the mesh file from msh to xdmf format"""
    pathlib.Path(OUTPUT).mkdir(parents=True, exist_ok=True)

    msh = OUTPUT / f"{GLOBAL_PARAMS['file_name']}.msh"
    return {
        "file_dep": [msh],
        "actions": [(convert_msh2xdmf, [msh])],
        "targets": [msh.with_suffix(".xdmf"), msh.with_suffix(".h5")],
        "clean": [clean_targets],
        "uptodate": [config_changed(GLOBAL_PARAMS)],
    }


@create_after(executed="convert_msh_xdmf")
def task_simulation():
    """run Fenics simulation"""
    # OUTPUT folder already existend

    xdmf = OUTPUT / f"{GLOBAL_PARAMS['file_name']}.xdmf"
    gcode = OUTPUT / f"{GLOBAL_PARAMS['file_name']}.gcode"
    xdmf_out = OUTPUT / f"{GLOBAL_PARAMS['file_name']}_out.xdmf"
    return {
        "file_dep": [xdmf, xdmf.with_suffix(".h5")],
        "actions": [(amsimulation, [GLOBAL_PARAMS, xdmf, gcode, xdmf_out])],
        "targets": [xdmf_out],
        "clean": [clean_targets],
        "uptodate": [config_changed(GLOBAL_PARAMS)],
    }
