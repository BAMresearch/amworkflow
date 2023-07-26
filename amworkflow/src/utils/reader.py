import os
from OCC.Extend.DataExchange import read_step_file, read_stl_file
from OCC.Core.TopoDS import TopoDS_Shape

def get_filename(path: str) -> str:
    return os.path.basename(path)

def step_reader(path: str) -> TopoDS_Shape():
    return read_step_file(filename=path)

def stl_reader(path: str) -> TopoDS_Shape:
    return read_stl_file(filename=path)