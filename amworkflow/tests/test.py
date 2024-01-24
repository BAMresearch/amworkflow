import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import re
from amworkflow.src.constants.data_model import MapParamModel
from amworkflow.src.utils.writer import namer
from amworkflow.src.geometries.simple_geometry import create_box
from OCC.Core.TopoDS import TopoDS_Solid, TopoDS_Face, TopoDS_Compound
from amworkflow.src.geometries.operator import intersector
from amworkflow.src.constants.data_model import DeepMapParamModel, MapParamModel
from amworkflow.src.utils.sanity_check import path_valid_check

# Writer
def is_hex(string):
    hex_pattern = r'^[0-9a-fA-F]+$'
    return re.match(hex_pattern, string) is not None and len(string) == 32

def test_data_model_with_init_data():
    label = ["a", "b", "c"]
    data = [1,2,3]
    model = MapParamModel(label=label, data=data)
    assert model.a ==1, "should be 1"
    
    
def test_data_model_without_init_data():
    label = ["a", "b", "c"]
    model = MapParamModel(label=label)
    model.a = 1
    assert model.a ==1, "should be 1"
    
def test_namer_hex():
    assert is_hex(namer(name_type="hex"))
    
def test_namer_dimension_and_batch():
    title = ["len", "height", "width"]
    data = [2.2, 3.1, 44]
    name = "L2_2-H3_1-W44-20230101"
    batch_num = 20230101
    assert namer(name_type="dimension-batch",dim_vector=data, parm_title=title), title
    
def test_name_mesh():
    assert namer(name_type="mesh", layer_param=0.2, geom_name="test"), "MeLT0.2-test"

def test_makebox():
    box = create_box(1,2,3)
    assert isinstance(box, TopoDS_Solid)
    
def test_intersector():
    com = intersector(create_box(1,2,3), 1.3, "z")
    assert isinstance(com, TopoDS_Face)
    
def test_dpdm():
    tdic = {"A":{"a":1},
        "B":{"b":2,
             "ba": 3},
        "C":{"c":{"ca":4}}}
    dmpm = DeepMapParamModel(tdic)
    assert dmpm.C.c.ca == 4
    
def test_pdm():
    lab = ["a", "b", "c"]
    val = [2, 3, 1]
    dat = MapParamModel(lab,val)
    assert dat.b == 3
