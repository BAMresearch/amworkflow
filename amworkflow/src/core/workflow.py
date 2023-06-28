from OCC.Core.TopoDS import TopoDS_Shape
import yaml
from src.constants.enums import Directory as D
from src.constants.data_model import WallParam, DB_WallGeometryFile, DB_XdmfFile, DB_H5File
from src.utils.parser import yaml_parser
import gmsh
from src.infrastructure.database.models.model import XdmfFile, H5File, FEResult, SliceFile, GCode
from src.infrastructure.database.cruds.crud import insert_data
from src.geometries.mesher import mesher, get_geom_pointer
from src.utils.writer import mesh_writer
from src.utils.permutator import simple_permutator
from src.utils.writer import namer, stl_writer, batch_num_creator
from tests.test import dimension_check
import numpy as np
from src.utils.download import downloader
import copy

class BaseWorkflow(object):
    def __init__(self, yaml_dir: str, filename: str, data_model: callable, geom_db_model: callable = None, geom_db_data_model: callable = None, db : bool = True):
        self.yaml_dir = yaml_dir
        self.yaml_filename = filename
        self.data_model = data_model
        self.yaml_parser = yaml_parser
        self.data = self.data_model(**yaml_parser(self.yaml_dir, self.yaml_filename))
        self.isbatch = self.data.batch_parameter.isbatch
        self.geom_pointer: int
        self.shape = []
        self.mesh_result = []
        self.name_list = []
        self.parm_list = []
        self.permutation: np.ndarray
        self.hashname_list = []
        self.mesh_name_list = []
        self.mesh_hashname_list = []
        self.db = db
        self.start_vector = []
        self.end_vector = []
        self.num_vector = []
        self.title = []
        self.db_data_collection = {}
        self.is_curve_list = []
        self.batch_num = None
        self.geom_db_model = geom_db_model
        self.geom_db_data_model = geom_db_data_model
        self.namer = namer
        pass
    
    def create(self) -> None:
        '''
        create the real entity of the geometry, then prepare necessary info for sequential process.
        Geometries created by this method should be placed in self.shape.
        '''
        self.data_assign()
        self.permutation = self.permutator()
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
    
    def data_assign(self):
        for ind, item in self.data.geometry_parameter:
            if item.length != None :
                self.start_vector.append(item.length)
            elif item.radius != None:
                self.start_vector.append(item.radius)
                self.is_curve_list.append(True)
            else:
                self.start_vector.append(0)
            self.end_vector.append(item.endpoint)
            self.num_vector.append(item.num)
            self.title.append(copy.copy(ind))
        self.start_vector = np.array(self.start_vector)
        self.end_vector = np.array(self.end_vector)
        self.num_vector = np.array(self.num_vector)
    
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