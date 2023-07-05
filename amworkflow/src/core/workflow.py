from OCC.Core.TopoDS import TopoDS_Shape
import yaml
import os
from amworkflow.src.constants.enums import Directory as D
from amworkflow.src.constants.enums import ParameterLabel as P
from amworkflow.src.constants.data_model import WallParam, DB_WallGeometryFile, DB_XdmfFile, DB_H5File
from amworkflow.src.utils.parser import yaml_parser
import gmsh
from amworkflow.src.infrastructure.database.models.model import XdmfFile, H5File, FEResult, SliceFile, GCode, ModelProfile, ModelParameter
from amworkflow.src.infrastructure.database.cruds.crud import insert_data, query_multi_data
from amworkflow.src.geometries.mesher import mesher, get_geom_pointer
from amworkflow.src.utils.writer import mesh_writer
from amworkflow.src.utils.permutator import simple_permutator
from amworkflow.src.utils.writer import namer, stl_writer, batch_num_creator
from amworkflow.tests.test import dimension_check
import numpy as np
from amworkflow.src.utils.download import downloader
from amworkflow.src.constants.data_model import MapParamModel
from amworkflow.src.utils.sanity_check import path_valid_check
import copy
from amworkflow.src.constants.exceptions import NoDataInDatabaseException, InsufficientDataException
from amworkflow.src.utils.reader import get_filename

class BaseWorkflow(object):
    def __init__(self, args):
        self.args = args
        self.yaml_dir = self.args.yaml_dir
        self.step_dir = self.args.step_dir
        self.yaml_parser = yaml_parser
        self.data: dict
        self.namer = namer
        self.model_name: str
        self.model_hashname = self.namer(name_type="hex")
        self.label = {}
        self.geom_pointer: int
        self.mpm = MapParamModel
        self.shape = []
        self.mesh_result = []
        self.name_list = []
        self.parm_list = []
        self.hashname_list = []
        self.mesh_name_list = []
        self.mesh_hashname_list = []
        self.db = True
        self.start_vector = []
        self.end_vector = []
        self.num_vector = []
        self.title = []
        self.db_data_collection = {}
        self.batch_num = None
        self.data_init()
    
    def data_init(self):
        if self.model_name != None:
            result = query_multi_data(ModelProfile, by_name=self.model_name, column_name="model_name", target_column_name="model_name")
            if self.model_name in result:
                indicator = (0,0)
                if self.args.edit:
                    indicator = (0,1)
                else:
                    if self.args.remove:
                        indicator = (0,2)
                    else:
                        indicator = (0,3)
            else:
                indicator = (0,4)
        else:    
            if self.step_dir != None:
                path_valid_check(self.step_dir, format=["stp", "step"])
                stp_filename = get_filename(self.step_dir)
                result = query_multi_data(ModelProfile, by_name=stp_filename, column_name="model_name", target_column_name="model_name")
                if stp_filename in result:
                    indicator = (1,0)
                else:
                    indicator = (1,1)
            else:
                if self.yaml_dir != None:
                    path_valid_check(self.yaml_dir, format=["yml", "yaml"])
                    self.data = yaml_parser(self.yaml_dir)
                    if "model_name" in self.data["model_profile"]:
                        self.model_name = self.data["model_profile"]["model_name"]
                        indicator = (2,0)
                    else: raise InsufficientDataException()
                else:
                    raise InsufficientDataException()
        
        match indicator[0]:
            case 1:
                #TODO: read the step file stored in the database and convert it to an OCC representation.
                match indicator[1]:
                        case 1:
                        #TODO: convert the file to an OCC representation and store the file and info to db.  
                            pass
            case 2:
                for lbl in self.data.keys():
                    if lbl == "geometry_parameter":
                        for key, item in self.data[lbl].items():
                            self.start_vector.append(item[P.STARTPOINT.value])
                            self.end_vector.append(item[P.ENDPOINT.value])
                            self.num_vector.append(item[P.NUM.value])
                            self.title.append(copy.copy(key))
                        self.start_vector = np.array(self.start_vector)
                        self.end_vector = np.array(self.end_vector)
                        self.num_vector = np.array(self.num_vector)
                        self.label[lbl] = self.title
                    else:
                        for key, item in self.data[lbl].items():
                            self.label[lbl].append(key)
                
            case 0:
                self.query_list = query_multi_data(table = ModelParameter,
                                    by_name= self.model_name,
                                    column_name="model_name")
                self.param_type = dict.fromkeys(set(rows["param_type"] for rows in self.query_list), {})
                self.param_list = [(rows["param_name"],rows["param_type"]) for rows in self.query_list]
                self.data = self.param_type.copy()
                for pair in self.param_list:
                    for p_type, p_value in self.data.items():
                        if pair[1] == p_type:
                            p_value[pair[0]] = None
                match indicator[1]:
                    case 1:
                        #TODO: compare differences between inputs and loaded parameters, replace and add new parameters.
                        pass
                    case 2:
                        #TODO: remove certain model profile from database
                        pass
                    case 3:
                        #TODO: do nothing, fill data into the loaded model.
                        pass
                    case 4:
                        #TODO: Create a new model profile with given parameters.
                        pass
                
    def create(self) -> None:
        '''
        create the real entity of the geometry, then prepare necessary info for sequential processes.
        Geometries created by this method should be placed in self.shape.
        '''
        dimension_check(self.permutation)  
        is_start_vector = False
        self.batch_num = batch_num_creator()
        self.db_data_collection["geometry"] = []
        for iter_ind, iter_perm in enumerate(self.permutation):
                if (np.linalg.norm(iter_perm - self.start_vector) == 0):
                    if is_start_vector == False:
                        self.geom_process(iter_ind, iter_perm, "dimension-batch")
                        is_start_vector = True
                    else:
                        pass
                else:
                    self.geom_process(iter_ind, iter_perm, "dimension-batch")
        if self.db:
            self.db_insert(db_model=self.geom_db_model, data=self.db_data_collection["geometry"])
            for ind, item in enumerate(self.shape):
                stl_writer(item=item,
                            item_name=self.hashname_list[ind] + ".stl",
                            linear_deflection= self.data.stl_parameter.linear_deflection,
                            angular_deflection= self.data.stl_parameter.angular_deflection)
                
    def geometry_spawn(self, param) -> TopoDS_Shape:
        '''Define a parameterized model using PyOCC APIs here with parameters defined in the yaml file. Return one single TopoDs_shape.'''
        return TopoDS_Shape
    
    def geom_process(self, ind: int, param: list, name_type: str):
        pass
    
    def mesh(self):
        '''
        mesh the geom created by create()
        '''   
        gmsh.initialize()
        self.db_data_collection["mesh"] = {"xdmf": [],
                                           "h5": []}
        for index, item in enumerate(self.shape):
            mesh_param = self.data.mesh_parameter
            is_thickness = mesh_param.layer_thickness.config
            size_factor = mesh_param.mesh_size_factor
            if is_thickness:
                layer_param = mesh_param.layer_thickness.num
            else:
                layer_param = mesh_param.layer_num.num
            model = mesher(item=item,
                   model_name=self.hashname_list[index],
                   layer_type = is_thickness,
                   layer_param=layer_param,
                   size_factor=size_factor)
            mesh_hashname = self.namer(name_type="hex")
            self.mesh_hashname_list.append(mesh_hashname)
            mesh_name = self.namer(name_type="mesh",
                                   is_layer_thickness=True,
                                   layer_param= layer_param,
                                   geom_name=self.name_list[index][:-4])
            self.mesh_name_list.append(mesh_name)
            if self.db:
                mesh_writer(item = model, 
                            directory=D.DATABASE_OUTPUT_FILE_PATH.value, 
                            filename=self.hashname_list[index],
                            output_filename = mesh_hashname,
                            format="xdmf")
                # mesh_writer(item=model,
                #             directory=D.USECASE_PATH_PARAMWALL_PATH.value,
                #             filename=self.hashname_list[index],
                #             output_filename = mesh_hashname,
                #             format="msh")
                db_model_xdmf = DB_XdmfFile()
                db_model_h5 = DB_H5File()
                db_model_xdmf.xdmf_hashname = mesh_hashname
                db_model_h5.h5_hashname = mesh_hashname
                db_model_h5.xdmf_hashname = mesh_hashname
                db_model_xdmf.filename = mesh_name + ".xdmf"
                db_model_h5.filename = mesh_name + ".h5"
                db_model_xdmf.mesh_size_factor = mesh_param.mesh_size_factor
                if is_thickness:
                    db_model_xdmf.layer_thickness = layer_param
                else:
                    db_model_xdmf.layer_num = layer_param
                db_model_xdmf.batch_num = self.batch_num
                db_model_xdmf.stl_hashname = self.hashname_list[index]
                db_model_h5.batch_num = self.batch_num
                xdmf_collection = self.db_data_collection["mesh"]["xdmf"]
                h5_collection = self.db_data_collection["mesh"]["h5"]
                xdmf_collection.append(db_model_xdmf.dict())
                h5_collection.append(db_model_h5.dict())
        gmsh.finalize()
        if self.db:
            self.db_insert(XdmfFile, xdmf_collection)
            self.db_insert(H5File, h5_collection)
        
    def slice():
        '''
        Slice the geometry created by create()
        '''
        pass
    
    def gcode():
        '''
        Create G-code for geometries sliced by slice()
        '''
        pass
    

    
    def permutator(self):
        _, perm = simple_permutator(start_point=self.start_vector,
                                 end_point=self.end_vector,
                                 num=self.num_vector)
        return perm

    def db_insert(self, db_model, data) -> None:
        insert_data(table=db_model, data=data, isbatch=self.isbatch)
    
    def upload(self):
        for ind, item in enumerate(self.shape):
            stl_writer(item=item,
                        item_name=self.hashname_list[ind] + ".stl",
                        linear_deflection= self.data.stl_parameter.linear_deflection,
                        angular_deflection= self.data.stl_parameter.angular_deflection)
            
    def download(self):
        if self.db:
            if self.batch_num != None:
                downloader(batch_num=self.batch_num)
                
    def item_namer(self, name_type, ind):
        hash_name = self.namer(name_type="hex")
        self.hashname_list.append(hash_name)
        filename = self.namer(name_type=name_type,
                    parm_title = self.title,
                    dim_vector=self.permutation[ind],
                    batch_num=self.batch_num) + ".stl"
        self.name_list.append(filename)
        return hash_name, filename