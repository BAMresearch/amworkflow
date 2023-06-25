import os
from src.infrastructure.database.cruds.crud import query_multi_data, delete_data
from src.infrastructure.database.models import model
import inspect
from src.constants.enums import InputFormat, Directory

def db_calib():
    tables = [model.GeometryFile]
    hashname_list = []
    file_lib = Directory.TEST_OUTPUT_PATH.value
    filename_list = [i.replace(".stl", "") for i in os.listdir(file_lib)]
    for table in tables:
        result = query_multi_data(table, None).all()
        for entry in result:
            hashname_list.append(entry[0].__dict__[InputFormat.STL_HASHNAME.value])
    unique_list = [i for i in hashname_list if i not in filename_list]
    delete_data(table, unique_list, True)
    
print(db_calib())
        