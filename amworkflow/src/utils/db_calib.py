import os
from amworkflow.src.infrastructure.database.cruds.crud import query_multi_data, delete_data
from amworkflow.src.infrastructure.database.models import model
import inspect
from amworkflow.src.constants.enums import Directory

def db_calib():
    tables = [model.GeometryFile]
    hashname_list = []
    file_lib = Directory.DATABASE_OUTPUT_FILE_PATH.value
    filename_list = [i.replace(".stl", "") for i in os.listdir(file_lib)]
    for table in tables:
        result = query_multi_data(table, None)
        for entry in result:
            hashname_list.append(entry[0].__dict__["stl_hashname"])
    unique_list = [i for i in hashname_list if i not in filename_list]
    delete_data(table, unique_list, True)
    
print(db_calib())
        