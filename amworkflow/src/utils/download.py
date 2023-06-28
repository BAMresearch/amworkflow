from src.infrastructure.database.cruds.crud import _query_data, query_multi_data
from src.infrastructure.database.models.model import GeometryFile
from src.constants.enums import Directory as D
from src.constants.enums import InputFormat as I
import shutil
import os

def downloader(batch_num: int | str = None,
               hashname: str = None,
               time_range: str = None):
    file_lib = D.TEST_OUTPUT_PATH.value
    if hashname != None:
        result = _query_data(GeometryFile, hashname).__dict__
        try:
            new_path = D.USECASE_PATH_PARAMWALL_PATH.value + result[I.STL_HASHNAME.value] + "/"
            if not os.path.exists(new_path):
                os.mkdir(new_path)
            copyer(file_lib + hashname + ".stl", 
                   dest_file = new_path  + result[I.FILE_NAME.value])
        except Exception as e:
            print(f"Unexpected error occurred: {e}")
    if batch_num != None:
        result = query_multi_data(GeometryFile, str(batch_num))
        new_path = D.USECASE_PATH_PARAMWALL_PATH.value + batch_num + "/"
        if not os.path.exists(new_path):
                os.mkdir(new_path)
        for row in result:
            _result = row[0].__dict__
            filename = _result[I.FILE_NAME.value]
            hash_name = _result[I.STL_HASHNAME.value]
            copyer(src_file= file_lib + hash_name + ".stl",
                   dest_file= new_path + filename)

def copyer(src_file: str, dest_file: str):
    shutil.copy2(src_file, dest_file)