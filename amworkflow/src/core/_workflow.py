#self dependencies
from amworkflow.src.interface.cli.cli_workflow import cli
from amworkflow.src.constants.data_model import DeepMapParamModel, MapParamModel
from amworkflow.src.constants.enums import Label as L
from amworkflow.src.utils.parser import geom_param_parser, yaml_parser, batch_data_parser
from amworkflow.src.utils.sanity_check import path_valid_check, dimension_check
from amworkflow.src.utils.permutator import simple_permutator
from amworkflow.src.constants.exceptions import NoDataInDatabaseException, InsufficientDataException
from amworkflow.src.interface.api import amWorkflow as aw
from amworkflow.src.utils.writer import task_id_creator
#3rd party lib dependencies
import numpy as np
import gmsh
#built-in dependencies
import copy
import subprocess
from multiprocessing import Pipe, Process

class BaseWorkflow(object):
    def __init__(self, args):
        self.args = args
        self.geometry_spawn: callable
        #signals
        self.indicator = self.task_handler()
        self.nodb = False
        self.isbatch = False
        self.isimport = False
        print(self.indicator)
        self.init_signal = self.data_init()
        # data
        self.geom_data = []
        self.pm = 0
        self.mdl = 0
        self.task_id = task_id_creator()
        self.task_dir = ""
        self.db_data_collect = {"geometry":[],
                                "occ": [],
                                "mesh":[]}
        #main
        self.sequence()
        
        
        
    def sequence(self):
        match self.init_signal:
            case 1:
                print("Done. Quitting...")
            case 0:
                self.geom_data = self.geom_param_handler()
                label = self.args.geom_param
                value = self.args.geom_param_value
                self.pm = MapParamModel(label, value)
                # self.create()
    
    def create(self) -> None:
        if self.isbatch:
            start_vec = np.array(self.args.geom_param_value)
            is_start_vector = False
            for iter_ind, iter_perm in enumerate(self.geom_data):
                if (np.linalg.norm(iter_perm - start_vec) == 0):
                    if is_start_vector == False:
                        self.process_geometry(param = iter_perm)
                        is_start_vector = True
                    else:
                        pass
                else:
                    self.process_geometry(iter_perm)
        else:
            self.process_geometry(param=self.args.geom_param_value)
        aw.db.insert_data("GeometryFile", self.db_data_collect["geometry"], True)
    def process_geometry(self, param: list):
        pm = MapParamModel(self.args.geom_param, param)
        occ = self.geometry_spawn(pm)
        hex = aw.tool.namer("hex")
        name = aw.tool.namer("dimension-batch", param, self.task_id, self.args.geom_param)
        data_in = {"filename": name,
                   "geom_hashname": hex,
                   "model_name": self.args.name,
                   "linear_deflection": self.args.stl_linear_deflect,
                   "angular_deflection": self.args.stl_angular_deflect,
                   "is_imported": self.isimport,
                   "task_id": self.task_id,
                   "stp": self.args.gen_stp}
        self.task_dir = aw.tool.mk_newdir(self.args.db_file_dir, self.task_id)
        aw.tool.write_stl(occ, hex, self.args.stl_linear_deflect, self.args.stl_angular_deflect, store_dir=self.task_dir)
        if self.args.gen_stp:
            aw.tool.write_step(occ, hex, self.task_dir)
        self.db_data_collect["geometry"].append(copy.copy(data_in))
        self.db_data_collect["occ"].append(occ)
        
    def mesh(self):
        is_thickness = True if self.args.mesh_by_thickness is not None else False
        size_factor = self.args.mesh_size_factor
        gmsh.initialize()
        g_info = self.db_data_collect["geometry"]
        for index, item in enumerate(self.db_data_collect["occ"]):
            if is_thickness:
                layer_param = self.args.mesh_by_thickness
            else:
                layer_param = self.args.mesh_by_layer
            model = aw.mesh.mesher(item=item,
                   model_name=g_info[index]["geom_hashname"],
                   layer_type = is_thickness,
                   layer_param=layer_param,
                   size_factor=size_factor)
            mesh_hashname = aw.tool.namer(name_type="hex")
            mesh_name = aw.tool.namer(name_type="mesh",
                                   is_layer_thickness=True,
                                   layer_param= layer_param,
                                   geom_name=g_info[index]["filename"])
            aw.tool.write_mesh(item = model, 
                        directory=self.task_dir, 
                        modelname=g_info[index]["geom_hashname"],
                        output_filename = mesh_hashname,
                        format="xdmf")
            if self.args.gen_msh:
                aw.tool.write_mesh(item = model, 
                        directory=self.task_dir, 
                        modelname=g_info[index]["geom_hashname"],
                        output_filename = mesh_hashname,
                        format="msh")
            if self.args.gen_vtk:
                aw.tool.write_mesh(item = model, 
                        directory=self.task_dir, 
                        modelname=g_info[index]["geom_hashname"],
                        output_filename = mesh_hashname,
                        format="vtk")    
            data_in = {"mesh_hashname":mesh_hashname,
                       "mesh_size_factor": self.args.mesh_size_factor,
                       "layer_thickness": self.args.mesh_by_thickness,
                       "layer_num": self.args.mesh_by_layer,
                       "task_id": self.task_id,
                       "geom_hashname": g_info[index]["filename"],
                       "msh": self.args.gen_msh,
                       "vtk": self.args.gen_vtk,
                       "filename": mesh_name}
            self.db_data_collect["mesh"].append(copy.copy(data_in))
        aw.db.insert_data("MeshFile",self.db_data_collect["mesh"], True)
        gmsh.finalize()
        
    def data_init(self):
        if self.args.stl_linear_deflect is None:
            self.args.stl_linear_deflect = 0.01
        if self.args.stl_angular_deflect is None:
            self.args.stl_angular_deflect = 0.1 
        match self.indicator[0]: 
            case 1: #read the imported file stored in the database and convert it to an OCC representation.
                md5 = aw.tool.get_md5(self.args.import_dir)
                impt_filename = aw.db.query_data("ImportedFile", by_name=md5, column_name="md5_id", only_for_column="filename")
                mdl_name = aw.db.query_data("ModelProfile", by_name=md5, column_name="imported_file_id", only_for_column="model_name")
                impt_format = path_valid_check(self.args.import_dir, format=["stp", "step","stl","STL"])
                if impt_format in ["stl","STL"]:
                    self.import_fl = aw.tool.read_stl(self.args.import_file_dir + "/" + impt_filename)
                else:
                    self.import_fl = aw.tool.read_step(self.args.import_file_dir + "/" + impt_filename)
                # if self.args.geom_param
                match self.indicator[1]:
                    case 1: # Import file, convert the file to an OCC representation and create a new profile for imported file. 
                        if impt_format in ["stl","STL"]:
                            self.import_fl = aw.tool.read_stl(self.args.import_dir)
                        else:
                            self.import_fl = aw.tool.read_step(self.args.import_dir)
                        
                    case 2: # remove selected profile and stp file, then quit 
                        aw.db.delete_data("ImportedFile", md5)
            # case 2: # yaml file provided
            #     # self.data = DeepMapParamModel(yaml_parser(self.args.yaml_dir))
            #     data = DeepMapParamModel(yaml_parser(self.args.yaml_dir))
            #     return data
                
            case 0: # model_name provided
                match self.indicator[1]:
                    case 1: #edit selected model
                        have_data, diff_new, diff_old, query = aw.db.have_data_in_db("ParameterToProfile", "param_name", self.args.geom_param, filter_by=self.args.name, search_column="model_name", search_column2="param_status", filter_by2=1)
                        new_p = []
                        active_p = []
                        if have_data:
                            print("No new parameters given, quitting...")
                        else:
                            for item in diff_old:
                                aw.db.update_data("ParameterToProfile",by_name = item, on_column="param_name", edit_column="param_status", new_value=False)
                            for item in diff_new:
                                q = aw.db.query_data("ModelParameter", by_name=item, column_name="param_name")
                                if q.empty:
                                    aw.db.insert_data("ModelParameter", {"param_name": item})
                                    aw.db.insert_data("ParameterToProfile", {
                                    "param_name": item,
                                    "model_name": self.args.name,
                                    "param_status": True})
                                    new_p.append(item)
                                else:
                                    ex_name = q["param_name"][0]
                                    aw.db.update_data("ParameterToProfile", ex_name,"param_name","param_status", True )
                                    active_p.append(item)
                            print(f"""Deactivate parameter(s):{diff_old} \nActivate parameter(s): {active_p+new_p} \nAdd new parameter(s): {new_p}""")        
                                    
                        if self.args.geom_param_value is not None:
                            self.indicator = (0,1,1)
                            
                    case 2:
                        #remove certain model profile from database
                        query_mdl_pfl = aw.db.query_data("ModelProfile", by_name=self.args.name, column_name="model_name")
                        q_md5 = query_mdl_pfl.imported_file_id[0]
                        if q_md5 != None:
                            aw.db.delete_data("ImportedFile", prim_ky=q_md5)
                        else:
                            aw.db.delete_data("ModelProfile", prim_ky=self.args.name)
                        print(f"model {self.args.name} has been removed successfully.")
                    case 0: # do nothing, fill data into the loaded model.
                        query = aw.db.query_data("ParameterToProfile", by_name=self.args.name, column_name="model_name", only_for_column="param_name", snd_column_name="param_status", snd_by_name=1)
                        if (self.args.geom_param is None) and (self.args.geom_param_value is None):
                            print(f"Activated Parameter(s) of model {self.args.name}: {query}")
                        if self.args.geom_param_value is not None:
                            self.indicator = (0,0,1)
                            self.args.geom_param = query
                    case 3: # Create a new model profile with given parameters.
                        aw.db.insert_data("ModelProfile", {"model_name": self.args.name})
                        if self.args.geom_param is not None:
                            input_data = {}
                            input_data2 = {}
                            collect = []
                            collect2 = []
                            have_data, diff,_,_ = aw.db.have_data_in_db("ModelParameter", "param_name", self.args.geom_param)
                            if not have_data:
                                for param in diff:
                                    input_data.update({"param_name": param})
                                    collect.append(copy.copy(input_data))
                                aw.db.insert_data("ModelParameter", collect, True)
                            for param in self.args.geom_param:
                                input_data2.update({"param_name": param,
                                                    "model_name": self.args.name})
                                collect2.append(copy.copy(input_data2))
                            aw.db.insert_data("ParameterToProfile", collect2, True)
                            
            case 3: # Draft mode.
                # temp_dir = aw.tool.mk_newdir(self.args.db_dir, "temp")
                pass
        if (self.indicator[2] == 1) and (self.args.geom_param_value is not None) and (self.args.geom_param is not None):
            return 0
        else:
            return 1
            
    
    def geom_param_handler(self):
        if self.args.iter_param is not None:
            end_vec, num_vec = batch_data_parser(len(self.args.geom_param), self.args.iter_param)
            start_vec = np.array(self.args.geom_param_value)
            permutation = simple_permutator(start_vec, end_vec, num_vec, self.args.geom_param)[1]
            self.isbatch = True
            return permutation
        else:
            return self.args.geom_param_value
    
    def task_handler(self):
        indicator = (0,0,0)
        match self.args.mode:
            case "draft":
                indicator = (3,0,0)
            case "production":
                if self.args.yaml_dir is not None:
                    path_valid_check(self.args.yaml_dir, format=["yml", "yaml"])
                    self.args = yaml_parser(self.args.yaml_dir)
                if self.args.name != None:
                    result = aw.db.query_data("ModelProfile", by_name=self.args.name, column_name=L.MDL_NAME.value, only_for_column=L.MDL_NAME.value)
                    if self.args.name in result:
                        indicator = (0,0,0)
                        if self.args.edit:
                            indicator = (0,1,0)
                        elif self.args.remove:
                            indicator = (0,2,0)   
                    elif self.args.import_dir != None:
                        impt_format = path_valid_check(self.args.import_dir, format=["stp", "step","stl","STL"])
                        self.isimport = True
                        impt_filename = aw.tool.get_filename(self.args.import_dir)
                        md5 = aw.tool.get_md5(self.args.import_dir)
                        result = aw.db.query_data("ImportedFile", by_name=md5, column_name="md5_id")
                        if not result.empty:
                            indicator = (1,0,0)
                            q_filename = result.filename[0]
                            print(f"Got same file {q_filename} in db, using existing file now...")
                            impt_filename = q_filename
                            if self.args.remove:
                                indicator = (1,2,0)
                        else:
                            aw.db.insert_data("ImportedFile",{"filename": impt_filename, "md5_id": md5})
                            aw.db.insert_data("ModelProfile", {"model_name": self.args.name,"imported_file_id": md5})
                            aw.tool.upload(self.args.import_dir, self.args.import_file_dir)
                            indicator = (1,1,0)
                    elif self.args.geom_param is not None:
                        indicator = (0,3,0)
                        if self.args.geom_param_value is not None:
                            indicator = (0,3,1)
                    else:
                        raise InsufficientDataException()
                # elif self.args.yaml_dir is not None:
                #     indicator = (2,0,0)
                else:
                    raise InsufficientDataException()
        return indicator
