import logging
import uuid
import os
import sys
import copy
from OCC.Core.StlAPI import StlAPI_Writer
from OCC.Extend.DataExchange import write_stl_file, write_step_file, read_stl_file
from tests.test import path_append_check
path_append_check()
from src.constants.enums import Directory
from src.constants.enums import Timestamp as T
from datetime import datetime
import numpy as np
import gmsh
from dolfinx.io import XDMFFile, gmshio
from mpi4py import MPI


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
            stl_output_dir = Directory.TEST_OUTPUT_PATH.value
            try:
                os.path.isdir(stl_output_dir)
            except:
                raise AssertionError("wrong path provided")
            write_stl_file(item,
                           stl_output_dir+item_name, mode="binary", linear_deflection = linear_deflection, 
                           angular_deflection = angular_deflection,
            )

def step_writer(item: any, filename: str):
    result = write_step_file(a_shape= item,
                             filename= filename)

def namer(name_type: str,
          dim_vector: np.ndarray = None,
          batch_num: int = None,
          parm_title: list = None,
          is_layer_thickness: bool = None,
          layer_param: float or int = None,
          geom_name: str = None
          ) -> str:
    if parm_title != None:
        title = [[j for j in i][0].upper() for i in parm_title]
    match name_type:
        case "hex":
            output = uuid.uuid4().hex
            
        case "dimension":
            dim_vector = [round(i, 3) for i in dim_vector]
            repl_vector = [str(i).replace(".", "_") for i in dim_vector]
            output = "-".join([title[i] + repl_vector[i] for i in range(len(title))])
            
        case "dimension-batch":
            dim_vector = [round(i, 3) for i in dim_vector]
            repl_vector = [str(i).replace(".", "_") for i in dim_vector]
            output = "-".join([title[i] + repl_vector[i] for i in range(len(title))]) + "-" + str(batch_num)
        
        case "mesh":
            if is_layer_thickness:
                output = f"MeLT_{layer_param}" + geom_name
            else:
                output = f"MeLN_{layer_param}" + geom_name
            
    return output

# print(namer("dimension", np.array([33.2, 22.44, 55.3, 66.8])))

def batch_num_creator():
        return datetime.now().strftime(T.YY_MM_DD_HH_MM_SS.value)

def vtk_writer(item: any,
               dirname:  str,
               filename: str) -> None:
    item.write(dirname + filename)

def mesh_writer(item: gmsh.model, directory: str, filename: str, output_filename: str, format: str):
    try:
        gmsh.is_initialized()
    except:
        logging.info("Gmsh must be initialized first!")
    item.set_current(filename)
    phy_gp = item.getPhysicalGroups()
    model_name = item.get_current()
    if format == "vtk":
        gmsh.write(directory + filename + format)
    if format == "xdmf":
        msh, cell_markers, facet_markers = gmshio.model_to_mesh(item, MPI.COMM_SELF, 0)
        msh.name = item.get_current()
        cell_markers.name = f"{msh.name}_cells"
        facet_markers.name = f"{msh.name}_facets"
        with XDMFFile(msh.comm, directory + f"{output_filename}.xdmf", "w") as file:
            file.write_mesh(msh)
            file.write_meshtags(cell_markers)
            msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
            file.write_meshtags(facet_markers)