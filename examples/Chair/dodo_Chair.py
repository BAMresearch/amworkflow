import datetime
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from doit import create_after, get_var
from doit.task import clean_targets
from doit.tools import config_changed

from fenicsxconcrete.util import ureg

from amworkflow.geometry import GeometryCenterline
from amworkflow.meshing import MeshingGmsh
from amworkflow.gcode import PowderbedCodeFromSTL
from amworkflow.simulation import SimulationFenicsXConcrete

# > doit -f <filename>   # for execution of all task
# > doit -f <filename> s <taskname> # for specific task
# > doit -f <filename> clean # for deleting task output

logging.basicConfig(level=logging.INFO)

# define required parameters
params_print = {  # print parameters
    #"stl_unit": 1000,       # units in your stl-file (how much of your unit is 1m, i.e. mm units would be factor 1000)
    #"debug_mode": False,    # if True, prints extra stuff into dsmn file, switch to false to create printable files 
    #"add_zeros": 0,         # additional zeros at the top and bottom of each layer in dsmn file             
}

# TODO datastore stuff??
OUTPUT_NAME = Path(__file__).parent.name
OUTPUT = (
    Path(__file__).parent / "output"
)  # / f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'


#@create_after(executed="create_design")
def task_print_instructions():
    """Generate printing instructions for BAM powder bed printer based on stl file."""
    OUTPUT.mkdir(parents=True, exist_ok=True)

    in_file_stl = "Smoothing_89_mm.stl"
    out_file_dsmn = OUTPUT / f"{OUTPUT_NAME}.dsmn"
    out_file_xyz = OUTPUT / f"{OUTPUT_NAME}.xyz"
    
    gcd = PowderbedCodeFromSTL(**params_print)

    return {
        "file_dep": [in_file_stl],
        "actions": [(gcd.create, [str(in_file_stl), str(out_file_dsmn), str(out_file_xyz)])],
        "targets": [out_file_dsmn, out_file_xyz],
        "clean": [clean_targets],
        "uptodate": [config_changed(params_print)],
    }