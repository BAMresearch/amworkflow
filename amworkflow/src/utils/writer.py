import logging
import uuid
import os
import sys
import copy
import shutil
from OCC.Core.StlAPI import StlAPI_Writer
from OCC.Extend.DataExchange import write_stl_file, write_step_file, read_stl_file
from amworkflow.src.constants.exceptions import GmshUseBeforeInitializedException
from amworkflow.src.constants.enums import Directory
from amworkflow.src.constants.enums import Timestamp as T
from datetime import datetime
import numpy as np
import gmsh
from dolfinx.io import XDMFFile, gmshio
from mpi4py import MPI

def stl_writer(item: any, item_name: str, linear_deflection: float = 0.001, angular_deflection: float = 0.1, output_mode = 1, store_dir: str = None) -> None:
    """
     @brief Write OCC to STL file. This function is used to write a file to the database. The file is written to a file named item_name. 
     @param item the item to be written to the file.
     @param item_name the name of the item. It is used to generate the file name.
     @param linear_deflection the linear deflection factor.
     @param angular_deflection the angular deflection factor.
     @param output_mode for using different api in occ.
     @param store_dir the directory to store the file in.
     @return None if success else error code ( 1 is returned if error ). In case of error it is possible to raise an exception
    """
    match output_mode:
        case 0:
            logging.info("Using stlAPI_Writer now...")
            stl_write = StlAPI_Writer()
            stl_write.SetASCIIMode(True)  # Set to False for binary STL output
            status = stl_write.Write(item, item_name)
            # if status is set to true then the status is not done.
            if status:
                logging.info("Done!")
        case 1:
            if linear_deflection is None:
                linear_deflection = 0.001
            if angular_deflection is None:
                angular_deflection = 0.1
            if item_name[-3:].lower() != "stl":
                item_name += ".stl"
            stl_output_dir = store_dir
            try:
                os.path.isdir(stl_output_dir)
            except:
                raise AssertionError("wrong path provided")
            write_stl_file(item,
                           os.path.join(stl_output_dir, item_name), mode="binary", linear_deflection = linear_deflection, 
                           angular_deflection = angular_deflection,
            )

def step_writer(item: any, filename: str, directory: str):
    """
     @brief Writes a step file. This is a wrapper around write_step_file to allow a user to specify the shape of the step and a filename
     @param item the item to write to the file
     @param filename the filename to write the file to ( default is None
    """
    try:
        os.path.isdir(directory)
    except:
        raise AssertionError("wrong path provided")
    if filename[-3:].lower() not in ["stl","stp", "tep"]:
        filename += ".stp"
    path = os.path.join(directory, filename)
    write_step_file(a_shape= item,
                             filename= path)

def namer(name_type: str,
          dim_vector: np.ndarray = None,
          task_id: int = None,
          parm_title: list = None,
          is_layer_thickness: bool = None,
          layer_param: float or int = None,
          geom_name: str = None
          ) -> str:
    """
           @brief Generate a name based on the type of name. It is used to generate an output name for a layer or a geometric object
           @param name_type Type of name to generate
           @param dim_vector Vector of dimension values ( default : None )
           @param task_id id of task to generate ( default : None )
           @param parm_title List of parameters for the layer
           @param is_layer_thickness True if the layer is thickness ( default : False )
           @param layer_param Parameter of the layer ( default : None )
           @param geom_name Name of the geometric object ( default : None )
           @return Name of the layer or geometric object ( default : None ) - The string representation of the nam
          """
    # Title of the parameter list.
    if parm_title != None:
        title = [[j for j in i][0].upper() for i in parm_title]
    # Replace. with _.
    if layer_param != None:
        layer_param = str(layer_param).replace(".", "_")
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
            output = "-".join([title[i] + repl_vector[i] for i in range(len(title))]) + "-" + str(task_id)
        
        case "mesh":
            # The layer thickness layer.
            if is_layer_thickness:
                output = f"MeLT{layer_param}-" + geom_name
            else:
                output = f"MeLN{layer_param}-" + geom_name
        case "only_import":
            output = geom_name + "_" + task_id
            
    return output

# print(namer("dimension", np.array([33.2, 22.44, 55.3, 66.8])))

def task_id_creator():
        """
         @brief Create task id for current date and time. It is used to determine how many batches are created in one batch.
         @return datetime. datetime date and time in YYYYMMDDHHMMSS format with format'%Y%m%d
        """
        return datetime.now().strftime(T.YY_MM_DD_HH_MM_SS.value)

def vtk_writer(item: any,
               dirname:  str,
               filename: str) -> None:
    """
    @brief Write a VTK file. This is a wrapper around the : py : func : ` ~voxel. write ` function that prepends the dirname and filename to the file name.
    @param item The item to write. It must be a
    @param dirname The directory to write the file to
    @param filename The filename to write to
    @return True if the write was successful False otherwise. >>> import numpy as np >>> vtk_writer = np. fromfile ( " test. vtk "
    """
    item.write(dirname + filename)

def mesh_writer(item: gmsh.model, directory: str, modelname: str, output_filename: str, format: str):
    """
     @brief Writes mesh to file. This function is used to write meshes to file. The format is determined by the value of the format parameter
     @param item gmsh. model object that contains the model
     @param directory directory where the file is located. It is the root of the file
     @param modelname name of the gmsh model to be written
     @param output_filename name of the file to be written
     @param format format of the file to be written. Valid values are vtk msh
    """
    try:
        gmsh.is_initialized()
    except:
        raise GmshUseBeforeInitializedException()
    item.set_current(modelname)
    phy_gp = item.getPhysicalGroups()
    model_name = item.get_current()
    # Write the format to the file.
    if format == "vtk" or format == "msh":
        name_with_f = output_filename + "." + format 
        new_dir = os.path.join(directory, name_with_f)
        gmsh.write(new_dir)
    # Create a mesh file for the current model.
    if format == "xdmf":
        msh, cell_markers, facet_markers = gmshio.model_to_mesh(item, MPI.COMM_SELF, 0)
        msh.name = item.get_current()
        cell_markers.name = f"{msh.name}_cells"
        facet_markers.name = f"{msh.name}_facets"
        name_with_f = output_filename + ".xdmf"
        new_dir = os.path.join(directory, name_with_f)
        with XDMFFile(msh.comm, new_dir, "w") as file:
            file.write_mesh(msh)
            file.write_meshtags(cell_markers)
            msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
            file.write_meshtags(facet_markers)

def mk_dir(dirname:str, folder_name: str):
    if os.path.isdir(dirname) == False:
        raise AssertionError("wrong path provided")
    newdir = dirname + "/" + folder_name
    if not os.path.exists(newdir):
        os.mkdir(newdir)
    return newdir


def convert_to_datetime(datetime_str):
    return datetime.strptime(datetime_str, "%Y%m%d%H%M%S")
