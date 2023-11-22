import os
from OCC.Extend.DataExchange import read_step_file, read_stl_file
from OCC.Core.TopoDS import TopoDS_Shape
import hashlib
import logging
import pandas as pd
import re


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

def is_md5_hash(s):
    md5_pattern = r"^[a-fA-F0-9]{32}$"
    return re.match(md5_pattern, s) is not None

import os
import logging
import ruamel.yaml

def yaml_parser(dir: str) -> dict:
    try:
        os.path.isdir(dir)
    except:
        logging.info("Wrong directory provided.")
    with open(dir, 'r') as yaml_file:
        yaml = ruamel.yaml.YAML(typ='safe')
        yaml.allow_duplicate_keys = True
        ipd = yaml.load(yaml_file)
        return ipd
