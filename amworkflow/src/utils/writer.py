import logging
import uuid
import os
import sys
import copy
from OCC.Core.StlAPI import StlAPI_Writer
from OCC.Extend.DataExchange import write_stl_file
from src.utils.sanity_check import path_append_check
path_append_check()
from src.constants.data_info_format import stl_info
from src.constants.enums import Directory
from src.constants.enums import Timestamp as T
from src.constants.enums import InputFormat as I
from datetime import datetime
import numpy as np


def stl_writer(item: any, item_name: str, linear_deflection: float = 0.001, angular_deflection: float = 0.1, output_mode = 1, store_dir: str = None) -> None:
    # if item_name == "hex":
    #     item_name = namer()
    match output_mode:
        case 0:
            logging.info("Using stlAPI_Writer to output now...")
            stl_write = StlAPI_Writer()
            stl_write.SetASCIIMode(True)  # Set to False for binary STL output
            status = stl_write.Write(item, item_name)
            if status:
                logging.info("Done!")
        case 1:
            stl_output_dir = Directory.SYS_PATH.value + "/stlOutput/"+"/testbatch/"
            try:
                os.path.isdir(stl_output_dir)
            except:
                raise AssertionError("wrong path provided")
            write_stl_file(item,
                           stl_output_dir+item_name, mode="binary", linear_deflection = linear_deflection, 
                           angular_deflection = angular_deflection,
            )

def namer(name_type: str,
          with_curve: bool = None,
          dim_vector: np.ndarray = None,
          batch_num: str = None
          ) -> str:
    if with_curve:
        title = ["L", "W", "H"]
    else:
        title = ["L", "W", "H", "R"]
    match name_type:
        case "hex":
            output = uuid.uuid4().hex
            
        case "dimension":
            dim_vector = [round(i, 3) for i in dim_vector]
            repl_vector = [str(i).replace(".", "_") for i in dim_vector]
            title = ["L", "W", "H", "R"]
            output = "-".join([title[i] + repl_vector[i] for i in range(len(title))])
            
        case "dimension-batch":
            dim_vector = [round(i, 3) for i in dim_vector]
            repl_vector = [str(i).replace(".", "_") for i in dim_vector]
            title = ["L", "W", "H", "R"]
            output = "-".join([title[i] + repl_vector[i] for i in range(len(title))]) + "-" + batch_num
            
    return output

# print(namer("dimension", np.array([33.2, 22.44, 55.3, 66.8])))

def batch_num_creator():
        return datetime.now().strftime(T.YY_MM_DD_HH_MM_SS.value)
    
def data_input(data: np.ndarray, input_type: str) -> dict:
    match input_type:
        case "stl":
            input_format = copy.copy(stl_info)
            input_format[I.WITH_CURVE.value] = data[0]
            input_format[I.LIN_DEFLECT.value] = data[1]
            input_format[I.ANG_DEFLECT.value] = data[2]
            input_format[I.BATCH_NUM.value] = data[3]
            input_format[I.LENGTH.value] = data[4]
            input_format[I.WIDTH.value] = data[5]
            input_format[I.HEIGHT.value] = data[6]
            input_format[I.RADIUS.value] = data[7]
            input_format[I.FILE_NAME.value] = data[8]
            input_format[I.STL_HASHNAME.value] = data[9]
            

    return input_format