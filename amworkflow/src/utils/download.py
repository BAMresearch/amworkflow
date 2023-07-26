from amworkflow.src.infrastructure.database.cruds.crud import _query_data, query_multi_data
from amworkflow.src.infrastructure.database.models.model import GeometryFile, XdmfFile, H5File
from amworkflow.src.constants.enums import Directory as D
import shutil
import os

def downloader(batch_num: int | str = None,
               hashname: str = None,
               time_range: str = None):
    file_lib = D.DATABASE_OUTPUT_FILE_PATH.value
    if hashname != None:
        result = _query_data(GeometryFile, hashname).__dict__
        new_folder = D.USECASE_PATH_PARAMWALL_PATH + result["filename"]
        if not os.path.exists(new_folder):
                os.mkdir(new_folder)
        try:
            new_path = D.USECASE_PATH_PARAMWALL_PATH + result["filename"] + "/"
            if not os.path.exists(new_path):
                os.mkdir(new_path)
            copyer(file_lib + hashname + ".stl", 
                   dest_file = new_path  + result["filename"])
        except Exception as e:
            print(f"Unexpected error occurred: {e}")
    if batch_num != None:
        result = query_multi_data(GeometryFile, str(batch_num), "batch_num")
        new_path = D.USECASE_PATH_PARAMWALL_PATH.value + batch_num + "/"
        if not os.path.exists(new_path):
                os.mkdir(new_path)
        for row in result:
            _result = row
            filename = _result["filename"]
            hash_name = _result["geom_hashname"]
            copyer(src_file= file_lib + hash_name + ".stl",
                   dest_file= new_path + filename)
            db_list = [XdmfFile, H5File]
            hash_list = ["xdmf_hashname", "h5_hashname"]
            format_list = [".xdmf", ".h5"]
            for ind, db_model in enumerate(db_list):
                try:
                    result = query_multi_data(db_model, str(batch_num))
                    for row in result:
                        _result = row[0].__dict__
                        filename = _result["filename"]
                        hash_name = _result[hash_list[ind]]
                        copyer(src_file= file_lib + hash_name + format_list[ind],
                            dest_file= new_path + filename)
                except Exception as e:
                    print(f"Unexpected error occurred: {e}")

def copyer(src_file: str, dest_file: str):
    shutil.copy2(src_file, dest_file)