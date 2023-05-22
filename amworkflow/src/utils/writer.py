import logging
import os
import sys
from OCC.Core.StlAPI import StlAPI_Writer
from OCC.Extend.DataExchange import write_stl_file
from src.utils.sanity_check import path_append_check
path_append_check()
from src.constants.enums import Directory


def stl_writer(item: any, item_name: str, linear_deflection: float = 0.5, angular_deflection: float = 0.3, output_mode = 1, store_dir: str = None) -> None:
    match output_mode:
        case 0:
            logging.info("Using stlAPI_Writer to output now...")
            stl_write = StlAPI_Writer()
            stl_write.SetASCIIMode(True)  # Set to False for binary STL output
            status = stl_write.Write(item, item_name)
            if status:
                logging.info("Done!")
        case 1:
            stl_output_dir = Directory.SYS_PATH.value + "/stlOutput/"
            try:
                os.path.isdir(stl_output_dir)
            except:
                raise AssertionError("wrong path provided")
            write_stl_file(item,
                           stl_output_dir+item_name, mode="binary", linear_deflection = linear_deflection, 
                           angular_deflection = angular_deflection,
            )
            
            
