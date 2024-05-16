import os
from pathlib import Path
import logging

from amworkflow.gcode.gcode import PowderbedCodeFromSTL

logging.basicConfig(level=logging.INFO)
# define required parameters
params = {
    "stl_unit": 1000,       # units in your stl-file (how much of your unit is 1m, i.e. mm units would be factor 1000)
    "debug_mode": False,    # if True, prints extra stuff into dsmn file, switch to false to create printable files 
    "add_zeros": 0,         # additional zeros at the top and bottom of each layer in dsmn file        
}

def test_gcodepowder(tmp_path):
    caller_path = Path(os.path.dirname(__file__))
    stl_in = caller_path / "cube_100x100x100mm.stl"
    dsmn_out = tmp_path / "cube_100x100x100mm.dsmn"
    xyz_out = tmp_path / "cube_100x100x100mm.xyz"
    gcd = PowderbedCodeFromSTL(**params)
    file_dsmn = Path(tmp_path) / "cube_100x100x100mm.dsmn"
    gcd.create(stl_in, dsmn_out, xyz_out)
    assert (file_dsmn.exists() & file_dsmn.stat().st_size > 0)