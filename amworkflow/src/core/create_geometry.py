from src.geometries.simple_geometry import create_box
import numpy as np
from src.constants.enums import Mapper as M
from src.utils.writer import stl_writer, namer, batch_num_creator, data_input
from src.utils.parser import yaml_parser
from src.utils.permutator import simple_permutator
from src.infrastructure.database.cruds.crud import insert_data
from src.infrastructure.database.models.model import STLFile

import copy

class CreateWall(object):
    def __init__(self, yaml_file_dir: str, yaml_file_name: str):
        self.l: float
        self.h: float
        self.w: float
        self.r: float
        self.isbatch: bool
        self.withcurve: bool
        self.l_ndp: float
        self.h_ndp: float
        self.w_ndp: float
        self.r_ndp: float
        self.l_num: float
        self.h_num: float
        self.w_num: float
        self.r_num: float
        self.batch_num: int
        self.yml_dir = yaml_file_dir
        self.yml_name = yaml_file_name
        self.lin_deflect: float
        self.ang_deflect: float
        self.hex_name: str
        self.hex_name_list: list
        self.db_data: list
    
    def data_assign(self) -> None:
        input_data = yaml_parser(self.yml_dir, self.yml_name)
        self.l = input_data[M.LENGTH.value]
        self.h = input_data[M.HEIGHT.value]
        self.w = input_data[M.WIDTH.value]
        self.r = input_data[M.RADIUS.value]
        self.start_vector = np.array([self.l, self.w, self.h, self.r]).astype(np.float64)
        self.isbatch = input_data[M.ISBATCH.value]
        self.withcurve = input_data[M.WITHCURVE.value]
        self.lin_deflect = input_data[M.LIN_DEFLECT.value]
        self.ang_deflect = input_data[M.ANG_DEFLECT.value]
        self.db_data = [self.withcurve, self.lin_deflect, self.ang_deflect]
        if self.isbatch == True:
            self.batch_num = batch_num_creator()
            self.l_ndp = input_data[M.L_ENDPOINT.value]
            self.h_ndp = input_data[M.H_ENDPOINT.value]
            self.w_ndp = input_data[M.W_ENDPOINT.value]
            self.r_ndp = input_data[M.R_ENDPOINT.value]
            self.l_num = input_data[M.L_NUM.value]
            self.h_num = input_data[M.H_NUM.value]
            self.w_num = input_data[M.W_NUM.value]
            self.r_num = input_data[M.R_NUM.value]
            self.num_vector = np.array([self.l_num, self.w_num, self.h_num, self.r_num])
            self.end_vector = np.array([self.l_ndp, self.w_ndp, self.h_ndp, self.r_ndp]).astype(np.float64)

    def create_wall(self):
        CreateWall.data_assign(self)
        if self.isbatch == True:
            geom_store = []
            name_store = []
            permutation = CreateWall.permutator(self)
            db_data_collection = []
            is_start_vector = False
            for iter_ind, iter_perm in enumerate(permutation):
                if (np.linalg.norm(iter_perm - self.start_vector) == 0):
                    if is_start_vector == False:
                        tp_box = create_box(length=iter_perm[0],
                                width=iter_perm[1],
                                height=iter_perm[2],
                                radius=iter_perm[3])
                        geom_store.append(tp_box)
                        name_store.append(np.copy(iter_perm)) 
                        is_start_vector = True
                    else:
                        pass
                else:
                    tp_box = create_box(length=iter_perm[0],
                                width=iter_perm[1],
                                height=iter_perm[2],
                                radius=iter_perm[3])
                    geom_store.append(tp_box)     
                    name_store.append(np.copy(iter_perm))         
            for ind, item in enumerate(geom_store):
                opt_name = namer(name_type="dimension-batch",
                                 with_curve=self.withcurve,
                                 dim_vector=name_store[ind],
                                 batch_num=self.batch_num) + ".stl"
                opt_name_hex = namer("hex")
                stl_writer(item=item,
                        item_name=opt_name_hex + ".stl",
                        linear_deflection= self.lin_deflect,
                        angular_deflection= self.ang_deflect)
                self.db_data.append(self.batch_num)
                self.db_data += np.ndarray.tolist(name_store[ind])
                self.db_data.append(opt_name)
                self.db_data.append(opt_name_hex)
                db_data_collection.append(copy.copy(self.db_data))
                CreateWall.db_data_renew(self)
        else:
            tp_box = create_box(length=self.l,
                                width=self.w,
                                height=self.h,
                                radius=self.r)
            opt_name = namer(name_type="dimension",
                             with_curve=self.withcurve,
                             dim_vector=self.start_vector) + ".stl"
            opt_name_hex = namer("hex")
            stl_writer(item=tp_box,
                       item_name=opt_name_hex + ".stl",
                       linear_deflection=self.lin_deflect,
                       angular_deflection=self.ang_deflect)
            self.db_data.append(self.batch_num)
            self.db_data += np.ndarray.tolist(self.start_vector)
            self.db_data.append(opt_name)
            self.db_data.append(opt_name_hex)
            db_data_collection = self.db_data
        CreateWall.db_insert(self, db_data_collection)
    
    def permutator(self):
        _, perm = simple_permutator(start_point=self.start_vector,
                                 end_point=self.end_vector,
                                 num=self.num_vector)
        return perm
    
    def db_insert(self, data: list) -> None:
        output = []
        if self.isbatch == True:
            for sub_perm in data:
                output.append(data_input(data=sub_perm,
                                         input_type="stl"))
            insert_data(STLFile, output, True)
        else:
            output = data_input(data=data, input_type="stl")
            insert_data(STLFile, output, False)
    
    def db_data_renew(self):
        self.db_data = [self.withcurve, self.lin_deflect,self.ang_deflect]
    

    
        
        
    