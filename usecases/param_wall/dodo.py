from param_wall import ParamWallWorkflow
from amworkflow.src.constants.data_model import WallGeometryParameter, DB_WallGeometryFile
from amworkflow.src.infrastructure.database.models.model import GeometryFile

def create_param_wall(yaml_dir: str, 
                      filename: str,
                      data_model: callable,
                      geom_db_data_model: callable,
                      geom_db_model: callable
                      ):
    wall = ParamWallWorkflow(yaml_dir=yaml_dir,
                             filename= filename,
                             data_model= data_model,
                             geom_db_data_model= geom_db_data_model,
                             geom_db_model=geom_db_model
                             )

def task_create():
    pass