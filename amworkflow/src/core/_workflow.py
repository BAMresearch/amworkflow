#self dependencies
from amworkflow.src.interface.cli.cli_workflow import cli
from amworkflow.src.constants.data_model import DeepMapParamModel
from amworkflow.src.constants.enums import Label as L
from amworkflow.src.utils.parser import geom_param_parser
from amworkflow.src.utils.sanity_check import path_valid_check, dimension_check
from amworkflow.src.constants.exceptions import NoDataInDatabaseException, InsufficientDataException
from amworkflow.src.interface.api import amWorkflow as aw
#pip lib dependencies
import numpy as np
#built-in dependencies
import copy

class BaseWorkflow(object):
    def __init__(self, args):
        self.args = args
        self.geometry_spawn: callable
        self.geom_param_handler()
        self.indicator = task_handler(self.args)
        print(self.indicator)
        self.data_init()
    def sequence(self):
        pass
    
    def create(self):
        pass
    
    def process_geometry(self):
        pass

    def data_init(self):
        match self.indicator[0]: 
            case 1: #read the imported file stored in the database and convert it to an OCC representation.
                md5 = aw.tool.get_md5(self.args.import_dir)
                impt_filename = aw.db.query_data("ImportedFile", by_name=md5, column_name="md5_id", only_for_column="filename")
                mdl_name = aw.db.query_data("ModelProfile", by_name=md5, column_name="imported_file_id", only_for_column="model_name")
                print(mdl_name)
                print(impt_filename)
                impt_format = path_valid_check(self.args.import_dir, format=["stp", "step","stl","STL"])
                if impt_format in ["stl","STL"]:
                    self.import_fl = aw.tool.read_stl(self.args.import_file_dir + "/" + impt_filename)
                else:
                    self.import_fl = aw.tool.read_step(self.args.import_file_dir + "/" + impt_filename)
                match self.indicator[1]:
                        case 1: # Import file, convert the file to an OCC representation and create a new profile for imported file. 
                            if impt_format in ["stl","STL"]:
                                self.import_fl = aw.tool.read_stl(self.args.import_dir)
                            else:
                                self.import_fl = aw.tool.read_step(self.args.import_dir)
                            
                        case 2: # remove selected profile and stp file, then quit
                            #TODO remove the step file and and info in db. 
                            aw.db.delete_data("ImportedFile", md5)
            case 2: # yaml file provided
                pass
                
            case 0: # model_name provided
                match self.indicator[1]:
                    case 2:
                        #remove certain model profile from database
                        query_mdl_pfl = aw.db.query_data("ModelProfile", by_name=self.args.name, column_name="model_name")
                        q_md5 = query_mdl_pfl.imported_file_id[0]
                        if q_md5 != None:
                            aw.db.delete_data("ImportedFile", prim_ky=q_md5)
                        else:
                            aw.db.delete_data("ModelProfile", prim_ky=self.args.name)
                    case 0: # do nothing, fill data into the loaded model.
                        pass
                    case 3: # Create a new model profile with given parameters.
                        aw.db.insert_data("ModelProfile", {"model_name": self.args.name})
                        if self.args.geom_param != None:
                            input = {"model_name": self.args.name}
                            collect = []
                            for param in self.args.geom_param:
                                print(param)
                                input.update({"param_name": param})
                                collect.append(copy.copy(input))
                            aw.db.insert_data("ModelParameter", collect, True)
            case 3: # Draft mode.
                pass
 
    def geom_param_handler(self):
        if self.args.geom_param != None:
            data = geom_param_parser(self.args)
            data.update({L.MESH_PARAM.value: {
                L.MESH_SIZE_FACTOR:self.args.mesh_size_factor, 
                L.LYR_NUM.value: self.args.mesh_by_layer, 
                L.LYR_TKN.value: self.args.mesh_by_layer}})
            data.update({L.STL_PARAM.value:{
                L.LNR_DFT.value: self.args.stl_linear_deflect,
                L.ANG_DFT.value: self.args.stl_angular_deflect
            }})
    
    
    
def task_handler(args):
    indicator = (0,0)
    match args.mode:
        case "draft":
            indicator = (3,0)
        case "production":
            if args.name != None:
                result = aw.db.query_data("ModelProfile", by_name=args.name, column_name=L.MDL_NAME.value, only_for_column=L.MDL_NAME.value)
                if args.name in result:
                    indicator = (0,0)
                    if args.edit:
                        indicator = (0,1)
                    elif args.remove:
                        indicator = (0,2)   
                elif args.import_dir != None:
                    impt_format = path_valid_check(args.import_dir, format=["stp", "step","stl","STL"])
                    impt_filename = aw.tool.get_filename(args.import_dir)
                    md5 = aw.tool.get_md5(args.import_dir)
                    result = aw.db.query_data("ImportedFile", by_name=md5, column_name="md5_id")
                    if not result.empty:
                        indicator = (1,0)
                        if args.remove:
                            indicator = (1,2)
                    else:
                        aw.db.insert_data("ImportedFile",{"filename": impt_filename, "md5_id": md5})
                        aw.db.insert_data("ModelProfile", {"model_name": args.name,"imported_file_id": md5})
                        aw.tool.upload(args.import_dir, args.import_file_dir)
                        indicator = (1,1)
                elif args.yaml_dir != None:
                    path_valid_check(args.yaml_dir, format=["yml", "yaml"])
                    indicator = (2,0)
                else:
                    indicator = (0,3)
            else:
                raise InsufficientDataException()
    return indicator
