from OCC.Core.TopoDS import TopoDS_Shape
import yaml
import os
from amworkflow.src.constants.enums import Directory as D
from amworkflow.src.constants.enums import Label as L
from amworkflow.src.utils.parser import yaml_parser, cmd_parser
import gmsh
from amworkflow.src.infrastructure.database.models.model import XdmfFile, H5File, FEResult, SliceFile, GCode, ModelProfile, ModelParameter
from amworkflow.src.infrastructure.database.cruds.crud import insert_data, query_multi_data, delete_data
from amworkflow.src.geometries.mesher import mesher, get_geom_pointer
from amworkflow.src.utils.writer import mesh_writer
from amworkflow.src.utils.permutator import simple_permutator
from amworkflow.src.utils.writer import namer, stl_writer, batch_num_creator
import numpy as np
from amworkflow.src.utils.download import downloader
from amworkflow.src.constants.data_model import MapParamModel
from amworkflow.src.utils.sanity_check import path_valid_check, dimension_check
from amworkflow.src.constants.exceptions import NoDataInDatabaseException, InsufficientDataException
from amworkflow.src.utils.reader import get_filename
from amworkflow.src.utils.reader import step_reader

class BaseWorkflow(object):
    def __init__(self, args):
        self.raw_args = args
        self.args = cmd_parser(args)
        self.yaml_dir = self.raw_args.yaml_dir
        self.step_dir = self.raw_args.step_dir
        self.yaml_parser = yaml_parser
        self.namer = namer
        self.model_name = self.raw_args.name
        self.isbatch = self.args[L.BATCH_PARAM.value][L.IS_BATCH.value]
        self.model_hashname = self.namer(name_type="hex")
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
        self.db_data_collection = {}
        self.db_del_collection = []
        self.batch_num = None
        self.data_init()
        self.permutation = self.permutator()
    
    def data_init(self):
        if self.model_name != None:
            result = query_multi_data(ModelProfile, by_name=self.model_name, column_name=L.MDL_NAME.value, target_column_name=L.MDL_NAME.value)
            if self.model_name in result:
                self.indicator = (0,0)
                if self.args.edit:
                    self.indicator = (0,1)
                else:
                    if self.args.remove:
                        self.indicator = (0,2)
                    else:
                        self.indicator = (0,3)
            else:
                self.indicator = (0,4)
        else:    
            if self.step_dir != None:
                path_valid_check(self.step_dir, format=["stp", "step"])
                self.stp_filename = get_filename(self.step_dir)
                result = query_multi_data(ModelProfile, by_name=self.stp_filename, column_name=L.MDL_NAME.value, target_column_name=L.MDL_NAME.value)
                if self.stp_filename in result:
                    self.indicator = (1,0)
                    if self.args.remove:
                        self.indicator = (1,2)
                else:
                    self.indicator = (1,1)
            else:
                if self.yaml_dir != None:
                    path_valid_check(self.yaml_dir, format=["yml", "yaml"])
                    self.data = yaml_parser(self.yaml_dir)
                    if L.MDL_NAME.value in self.data["model_profile"]:
                        self.model_name = self.data["model_profile"][L.MDL_NAME.value]
                        self.indicator = (2,0)
                    else: raise InsufficientDataException()
                else:
                    raise InsufficientDataException()
        
        match self.indicator[0]: 
            case 1: #read the step file stored in the database and convert it to an OCC representation.
                self.import_model = step_reader(path=D.DATABASE_OUTPUT_FILE_PATH.value+self.stp_filename)
                match self.indicator[1]:
                        case 1: # Import stp file, convert the file to an OCC representation and create a new profile for imported stp file. 
                            self.import_model = step_reader(self.step_dir)
                            self.db_data_collection[L.MDL_PROF.value] = {L.MDL_NAME.value: self.stp_filename} 
                        case 2: # remove selected profile and stp file, then quit
                            #TODO remove the step file and and info in db. 
                            pass
            case 2: # yaml file provided
                self.geom_data = self.data[L.GEOM_PARAM.value]
                self.batch_data_convert(data=self.geom_data)
                self.param_type = self.geom_data.keys()
                self.mesh_param = self.data[L.MESH_PARAM.value]
                
            case 0: # model_name provided
                self.query_list = query_multi_data(table = ModelParameter,
                                    by_name= self.model_name,
                                    column_name=L.MDL_NAME.value)
                self.param_type = list(set(rows[L.PARAM_TP.value] for rows in self.query_list))
                if self.indicator[1] == 1: # edit mode. edit profile, replace the old and continue with new parameters.
                    self.db_del_collection.append(ModelParameter,self.param_type)
                    self.db_delete()
                    self.param_type = self.args[L.GEOM_PARAM.value].keys()
                    value = [{L.PARAM_NAME.value: new_param[i],
                                  L.MDL_NAME.value: self.stp_filename} for i in range(len(self.param_type))]
                    self.db_insert(ModelParameter, value)
                match self.indicator[1]:
                    case 2:
                        #remove certain model profile from database
                        self.db_del_collection.append(ModelProfile, [self.model_name])
                        self.db_delete()
                    case 3: # do nothing, fill data into the loaded model.
                        self.param_type = self.args[L.GEOM_PARAM.value].keys()
                    case 4: # Create a new model profile with given parameters.
                        self.db_data_collection[L.MDL_PROF.value] = [{L.MDL_NAME.value: self.raw_args.name}]
                        new_param = list(self.args[L.GEOM_PARAM.value].keys())
                        print(new_param)
                        value = [{L.PARAM_NAME.value: new_param[i],
                                  L.MDL_NAME.value: self.model_name} for i in range(len(new_param))]
                        self.db_data_collection[L.MDL_PARAM.value] = value
                        self.param_type = self.args[L.GEOM_PARAM.value].keys()
                        self.db_insert(ModelProfile, self.db_data_collection[L.MDL_PROF.value])
                        self.db_insert(ModelParameter, self.db_data_collection[L.MDL_PARAM.value])
                self.batch_data_convert(data=self.args[L.GEOM_PARAM.value])
                
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
        param = self.mpm(param, self.param_type)
        tp_geom = self.geometry_spawn(param)
        self.shape.append(tp_geom)
        hash_name, filename = self.item_namer(name_type=name_type, ind=ind)
        append_data = {"batch_num" : self.batch_num,
                       "geom_hashname" : hash_name,
                       "filename" : filename}
        self.db_data_collection["geometry"].append(append_data)
    
    def mesh(self):
        '''
        mesh the geom created by create()
        '''   
        if self.indicator == (2,0):
            data=self.mesh_param.values()
            label=self.mesh_param.keys()
        else:
            data=self.args[L.MESH_PARAM.value].values()
            label=self.args[L.MESH_PARAM.value].keys()
        mesh_param = self.mpm(data=data, label=label)
        gmsh.initialize()
        self.db_data_collection["mesh"] = {"xdmf": [],
                                           "h5": []}
        for index, item in enumerate(self.shape):
            is_thickness = True if L.LYR_TKN != None in self.raw_args[L.LYR_TKN.value] else False
            size_factor = mesh_param.mesh_size_factor
            if is_thickness:
                layer_param = mesh_param.layer_thickness
            else:
                layer_param = mesh_param.layer_num
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
                xdmf_label = ["xdmf_hashname", "filename", "mesh_size_factor", "layer_thickness", "layer_num", "batch_num","stl_hashname"]
                h5_label = ["5_hashname", "filename", "batch_num"]
                db_model_xdmf = self.mpm(xdmf_label)
                db_model_h5 = self.mpm(h5_label)
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
                xdmf_collection.append(db_model_xdmf.dict)
                h5_collection.append(db_model_h5.dict)
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
        
    def db_delete(self) -> None:
        for request in self.db_del_collection:
            table = request[0]
            delete_data(table=table, by_primary_key=request[1], isbatch=self.isbatch)
    
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
    
    def batch_data_convert(self, data: dict) -> None:
        for key, item in data.items():
            self.start_vector.append(item[L.STARTPOINT.value])
            self.end_vector.append(item[L.ENDPOINT.value])
            self.num_vector.append(item[L.NUM.value])
        self.start_vector = np.array(self.start_vector)
        self.end_vector = np.array(self.end_vector)
        self.num_vector = np.array(self.num_vector)