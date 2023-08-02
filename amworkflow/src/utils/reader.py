import os
from OCC.Extend.DataExchange import read_step_file, read_stl_file
from OCC.Core.TopoDS import TopoDS_Shape
import hashlib
import logging
from amworkflow.src.infrastructure.database.cruds.crud import query_multi_data
import pandas as pd


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
    
def having_data(table: str, column_name: str, dataset: list, search_column: str = None, filter: str = None) -> bool | list:
    result = query_multi_data(table=table, by_name=filter, column_name=search_column)
    if isinstance(result, pd.DataFrame) and not result.empty:
        query_list = result[column_name].to_list()
        diff = list(set(dataset) - set(query_list))
        diff2 = list(set(query_list) - set(dataset))
        if len(diff) + len(diff2)== 0:
            return True, 0, 0, 0
        else:
            return False, diff, diff2, query_list
    else:
        return False, dataset, 0, 0
