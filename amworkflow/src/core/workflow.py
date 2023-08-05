from OCC.Core.TopoDS import TopoDS_Shape
import sys
import os
from amworkflow.src.constants.enums import Directory as D
from amworkflow.src.constants.enums import Label as L
from amworkflow.src.utils.parser import yaml_parser, cmd_parser
import gmsh
import amworkflow.src.infrastructure.database.engine.config as CG
from amworkflow.src.infrastructure.database.models.model import XdmfFile, H5File, FEResult, SliceFile, GCode, ModelProfile, ModelParameter, GeometryFile
from amworkflow.src.infrastructure.database.cruds.crud import insert_data, query_multi_data, delete_data
from amworkflow.src.geometries.mesher import mesher
from amworkflow.src.utils.writer import mesh_writer, mk_dir
from amworkflow.src.utils.permutator import simple_permutator
from amworkflow.src.utils.writer import namer, stl_writer, batch_num_creator
import numpy as np
from amworkflow.src.utils.db_io import downloader
from amworkflow.src.constants.data_model import MapParamModel, DeepMapParamModel
from amworkflow.src.utils.sanity_check import path_valid_check, dimension_check
from amworkflow.src.constants.exceptions import NoDataInDatabaseException, InsufficientDataException
from amworkflow.src.utils.reader import get_filename
from amworkflow.src.utils.reader import step_reader, stl_reader
from amworkflow.src.interface.cli.cli_workflow import cli
DB_FILES_DIR = ""

class BaseWorkflow(object):
    def __init__(self, args):
        self.raw_args = args
        self.geometry_spawn: callable
        print(self.raw_args)
        print(os.getcwd())
        self.mpm = MapParamModel
        self.dmpm = DeepMapParamModel
        self.parsed_args = cmd_parser(self.raw_args)
        self.parsed_args_c = self.dmpm(self.parsed_args)
        self.yaml_dir = self.raw_args.yaml_dir
        self.import_dir = self.raw_args.import_dir
        self.yaml_parser = yaml_parser
        self.namer = namer
        self.model_name = self.raw_args.name
        self.model_hashname = self.namer(name_type="hex")
        self.geom_pointer: int
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
        self.batch_num = self.parsed_args
        self.isbatch = False
        self.linear_deflect = 0.001
        self.angular_deflect = 0.1
        self.mesh_t = None
        self.mesh_n = None
        self.mesh_s = None
        self.indicator = task_handler(args=self.raw_args)
        self.data_init()
        self.permutation = self.permutator()
                
    def data_init(self):
        match self.indicator[0]: 
            case 1: #read the step file stored in the database and convert it to an OCC representation.
                self.linear_deflect = self.raw_args.stl_linear_deflect
                self.angular_deflect= self.raw_args.stl_angular_deflect
                self.mesh_t = self.raw_args.mesh_by_thickness
                self.mesh_n = self.raw_args.mesh_by_layer
                self.mesh_s = self.raw_args.mesh_size_factor
                self.isbatch = self.parsed_args[L.BATCH_PARAM.value][L.IS_BATCH.value]
                impt_format = path_valid_check(self.raw_args.import_dir, format=["stp", "step","stl","STL"])
                impt_filename = get_filename(self.raw_args.import_dir)
                match impt_format.lower():
                    case "stl":
                        self.import_model = stl_reader(path=self.import_dir)
                    case "stp":
                        self.import_model = step_reader(path=self.import_dir)
                    case "step":
                        self.import_model = step_reader(path=D.DATABASE_OUTPUT_FILE_PATH.value+impt_filename)
                match self.indicator[1]:
                        case 1: # Import file, convert the file to an OCC representation and create a new profile for imported file. 
                            self.db_data_collection[L.MDL_PROF.value] = {L.MDL_NAME.value: impt_filename} 
                            # self.batch_data_convert(self.parsed_args[L.GEOM_PARAM.value])
                        case 2: # remove selected profile and stp file, then quit
                            #TODO remove the step file and and info in db. 
                            pass
            case 2: # yaml file provided
                self.data = yaml_parser(self.yaml_dir)
                if L.MDL_NAME.value in self.data["model_profile"]:
                    self.model_name = self.data["model_profile"][L.MDL_NAME.value]
                else: raise InsufficientDataException()
                self.geom_data = self.data[L.GEOM_PARAM.value]
                self.batch_data_convert(data=self.geom_data)
                self.param_type = self.geom_data.keys()
                self.mesh_param = self.data[L.MESH_PARAM.value]
                self.linear_deflect = self.data[L.STL_PARAM.value][L.LNR_DFT.value]
                self.angular_deflect = self.data[L.STL_PARAM.value][L.ANG_DFT.value]
                self.mesh_t = self.data[L.MESH_PARAM.value][L.LYR_TKN.value]
                self.mesh_n = self.data[L.MESH_PARAM.value][L.LYR_NUM.value]
                self.mesh_s = self.data[L.MESH_PARAM.value][L.MESH_SIZE_FACTOR.value]
                if sum(self.num_vector) > 1:
                    self.isbatch = True
                    self.batch_data_convert(self.parsed_args[L.GEOM_PARAM.value])
                else:
                    self.isbatch = False
                
            case 0: # model_name provided
                self.linear_deflect = self.raw_args.stl_linear_deflect
                self.angular_deflect= self.raw_args.stl_angular_deflect
                self.mesh_t = self.raw_args.mesh_by_thickness
                self.mesh_n = self.raw_args.mesh_by_layer
                self.mesh_s = self.raw_args.mesh_size_factor
                self.isbatch = self.parsed_args[L.BATCH_PARAM.value][L.IS_BATCH.value]
                self.query_list = query_multi_data(table = ModelParameter,
                                    by_name= self.model_name,
                                    column_name=L.MDL_NAME.value)
                self.param_type = list(set(rows[L.PARAM_TP.value] for rows in self.query_list))
                if self.indicator[1] == 1: # edit mode. edit profile, replace the old and continue with new parameters.
                    self.db_del_collection.append(ModelParameter,self.param_type)
                    self.db_delete()
                    self.param_type = self.parsed_args[L.GEOM_PARAM.value].keys()
                    value = [{L.PARAM_NAME.value: new_param[i],
                                  L.MDL_NAME.value: self.impt_filename} for i in range(len(self.param_type))]
                    self.db_insert(ModelParameter, value)
                match self.indicator[1]:
                    case 2:
                        #remove certain model profile from database
                        self.db_del_collection.append(ModelProfile, [self.model_name])
                        self.db_delete()
                    case 3: # do nothing, fill data into the loaded model.
                        self.param_type = self.parsed_args[L.GEOM_PARAM.value].keys()
                    case 4: # Create a new model profile with given parameters.
                        self.db_data_collection[L.MDL_PROF.value] = [{L.MDL_NAME.value: self.raw_args.name}]
                        new_param = list(self.parsed_args[L.GEOM_PARAM.value].keys())
                        print(new_param)
                        value = [{L.PARAM_NAME.value: new_param[i],
                                  L.MDL_NAME.value: self.model_name} for i in range(len(new_param))]
                        self.db_data_collection[L.MDL_PARAM.value] = value
                        self.param_type = list(self.parsed_args[L.GEOM_PARAM.value].keys())
                        self.db_insert(ModelProfile, self.db_data_collection[L.MDL_PROF.value])
                        self.db_insert(ModelParameter, self.db_data_collection[L.MDL_PARAM.value])
        self.batch_data_convert(data=self.parsed_args[L.GEOM_PARAM.value])
                
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
            self.db_insert(db_model=GeometryFile, data=self.db_data_collection["geometry"])
            for ind, item in enumerate(self.shape):
                print(self.angular_deflect)
                stl_writer(item=item,
                            item_name=self.hashname_list[ind] + ".stl",
                            linear_deflection= self.linear_deflect,
                            angular_deflection= self.angular_deflect,
                            store_dir=DB_FILES_DIR)
    
    def geom_process(self, ind: int, param: list, name_type: str):
        param = self.mpm(self.param_type, param)
        tp_geom = self.geometry_spawn(param)
        self.shape.append(tp_geom)
        hash_name, filename = self.item_namer(name_type=name_type, ind=ind)
        append_data = {"batch_num" : self.batch_num,
                       "geom_hashname" : hash_name,
                       "filename" : filename,
                       "model_name": self.model_name}
        self.db_data_collection["geometry"].append(append_data)
    
    def mesh(self):
        '''
        mesh the geom created by create()
        '''   
        if self.indicator == (2,0):
            data=[self.mesh_n, self.mesh_t, self.mesh_s]
            label=list(self.mesh_param.keys())
        else:
            data=[self.mesh_n, self.mesh_t, self.mesh_s]
            label=[L.LYR_NUM.value,L.LYR_TKN.value,L.MESH_SIZE_FACTOR.value]
        mesh_param = self.mpm(data=data, label=label)
        print(mesh_param.dict)
        print(mesh_param.layer_thickness)
        is_thickness = True if mesh_param.layer_thickness != None else False
        size_factor = mesh_param.mesh_size_factor
        gmsh.initialize()
        self.db_data_collection["mesh"] = {"xdmf": [],
                                           "h5": []}
        for index, item in enumerate(self.shape):
            
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
                            directory=DB_FILES_DIR, 
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
                    parm_title = self.param_type,
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
    
    def create_database_engine(self):
        pass

def task_handler(args):
    if args.name != None:
        result = query_multi_data(ModelProfile, by_name=args.name, column_name=L.MDL_NAME.value, target_column_name=L.MDL_NAME.value)
        if args.name in result:
            indicator = (0,0)
            if args.edit:
                indicator = (0,1)
            else:
                if args.remove:
                    indicator = (0,2)
                else:
                    indicator = (0,3)
        else:
            indicator = (0,4)
    else:    
        if args.import_dir != None:
            impt_format = path_valid_check(args.import_dir, format=["stp", "step","stl","STL"])
            impt_filename = get_filename(args.import_dir)
            result = query_multi_data(ModelProfile, by_name=impt_filename, column_name=L.MDL_NAME.value, target_column_name=L.MDL_NAME.value)
            if impt_filename in result:
                indicator = (1,0)
                if args.remove:
                    indicator = (1,2)
            else:
                indicator = (1,1)
        else:
            if args.yaml_dir != None:
                path_valid_check(args.yaml_dir, format=["yml", "yaml"])
                indicator = (2,0)
            else:
                raise InsufficientDataException()
    return indicator