from pathlib import Path

from doit import create_after, get_var
from doit.task import clean_targets
from doit.tools import config_changed

# from amworkflow.geometry import ...

# > doit    # for execution of all task
# > doit s <taskname> # for specific task
# > doit clean # for deleting task output

params = {...
}

# TODO datastore stuff??
OUTPUT_NAME = Path(__file__).parent.name
OUTPUT = (
    Path(__file__).parent / "output"
)  # / f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

def task_create_design():
    """create the design

        if a step file was generated externally you can skip this task

        choose a geometry class or create a new one
        - centerline model -> geometryCenterline with given file where the center line point are stored
        - wall model -> geometryWall with parameters for length, height, width, radius, fill-in
        - new geometry -> create a new geometry class from geometry_base class and change the geometry_spawn method accordingly
    """
    OUTPUT.mkdir(parents=True, exist_ok=True)

    out_file_step = OUTPUT / f"{OUTPUT_NAME}.stp"
    out_file_stl = OUTPUT / f"{OUTPUT_NAME}.stl"
    out_file_points = OUTPUT / f"{OUTPUT_NAME}.csv"

    # geometry = ...(params)

    return {
        "actions": [(geometry.create,[])],
        "targets": [out_file_step, out_file_stl, out_file_points],
        "clean": [clean_targets],
        "uptodate": [config_changed(params)],
    }


@create_after(executed="create_design")
def task_meshing():
    """meshing a given design from a step file

        choose a meshing class or create a new one
        required parameters in params:
        - mesh_size: size of the mesh
        - meshing via number of layers or layer height possible
    """

    OUTPUT.mkdir(parents=True, exist_ok=True)

    in_file_step = OUTPUT / f"{OUTPUT_NAME}.stp"
    out_file_xdmf = OUTPUT / f"{OUTPUT_NAME}.xdmf"
    out_file_vtk = OUTPUT / f"{OUTPUT_NAME}.vtk"

    # new_meshing = ...(params)

    return {
        "file_dep": [in_file],
        "actions": [(new_meshing.create, [in_file_step, out_file_xdmf, out_file_vtk])],
        "targets": [out_file_xdmf, out_file_vtk],
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

    in_file_xdmf = OUTPUT / f"{OUTPUT_NAME}.xdmf"
    out_file_xdmf = OUTPUT / f"{OUTPUT_NAME}_sim.xdmf"

    # params_sim["experiment_type"] = "structure" * ureg("") # or "process"
    # simulation = SimulationFenicsXConcrete(params_sim)

    return {
        "file_dep": [in_file],
        "actions": [(simulation.run, [in_file_xdmf, out_file_xdmf])],
        "targets": [out_file_xdmf],
        "clean": [clean_targets],
        "uptodate": [config_changed(params)],
    }

    pass

@create_after(executed="create_design")
def task_gcode():
    """generate gcode from stl  or csv point file"""

    OUTPUT.mkdir(parents=True, exist_ok=True)

    in_file_points = OUTPUT / f"{OUTPUT_NAME}.csv"
    out_file_gcode = OUTPUT / f"{OUTPUT_NAME}.gcode"

    #gcd = GcodeFromPoints(**params)

    return {
        "file_dep": [in_file_points],
        "actions": [(gcd.create, [in_file_points, out_file_gcode])],
        "targets": [out_file_gcode],
        "clean": [clean_targets],
        "uptodate": [config_changed(params)],
    }

    pass






