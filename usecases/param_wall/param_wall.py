import sys
import os
import pathlib
#sys.path.append("/home/yhe/Documents/amworkflow/amworkflow")
from OCC.Core.TopoDS import TopoDS_Shape
from amworkflow.src.core.workflow import BaseWorkflow
from amworkflow.src.constants.data_model import WallParam, DB_WallGeometryFile
from amworkflow.src.infrastructure.database.models.model import GeometryFile
from amworkflow.src.geometries.simple_geometry import create_box

class ParamWallWorkflow(BaseWorkflow):
    def __init__(self, yaml_dir: str, filename: str, data_model: callable, geom_db_model: callable = None, geom_db_data_model: callable = None):
        super().__init__(yaml_dir, filename, data_model, geom_db_model, geom_db_data_model)
        self.gp = self.data.geometry_parameter

    def geometry_spawn(self, param) -> TopoDS_Shape:
        box = create_box(length=param[0],
                         width= param[2],
                         height=param[1],
                         radius=param[3])
        return box

    def geom_process(self, ind: int, param: list, name_type: str):
        tp_geom = self.geometry_spawn(param)
        self.shape.append(tp_geom)
        hash_name, filename = self.item_namer(name_type=name_type, ind=ind)
        model = self.geom_db_data_model()
        model.batch_num = self.batch_num
        model.withCurve = True if param[3] != 0 else False
        model.length = param[0]
        model.height = param[1]
        model.width = param[2]
        model.radius = param[3]
        model.linear_deflection = self.data.stl_parameter.linear_deflection
        model.angular_deflection = self.data.stl_parameter.angular_deflection
        model.stl_hashname = hash_name
        model.filename = filename
        model.stl_hashname = hash_name
        self.db_data_collection["geometry"].append(model.dict())

yaml_dir = pathlib.Path(__file__).parent
yaml_filename = "test1.yaml"

p_wall = ParamWallWorkflow(yaml_dir=str(yaml_dir),
                         filename=yaml_filename,
                         data_model= WallParam, 
                         geom_db_model= GeometryFile,
                         geom_db_data_model= DB_WallGeometryFile)
p_wall.create()
p_wall.mesh()
p_wall.download()