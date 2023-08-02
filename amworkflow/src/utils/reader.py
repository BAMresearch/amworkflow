import os
from OCC.Extend.DataExchange import read_step_file, read_stl_file
from OCC.Core.TopoDS import TopoDS_Shape
import hashlib
import logging

def get_filename(path: str) -> str:
    return os.path.basename(path)

def step_reader(path: str) -> TopoDS_Shape():
    return read_step_file(filename=path)

def stl_reader(path: str) -> TopoDS_Shape:
    return read_stl_file(filename=path)

def get_file_md5(path: str) -> str:
    with open(path, "rb") as file:
        data = file.read()
        md5 = hashlib.md5(data).hexdigest()
    return md5

def verification(path1: str, path2: str) -> bool:
    filename1 = get_filename(path1)
    filename2 = get_filename(path2)
    md51 = get_file_md5(path1)
    md52 = get_file_md5(path2)
    if md51 == md52:
        logging.info(f"{filename1} and {filename2} are identical.")
        return True
    else:
        return False
    