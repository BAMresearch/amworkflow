from amworkflow.src.infrastructure.database.cruds.crud import query_multi_data, update_data, query_data_object, delete_data
import amworkflow.src.infrastructure.database.engine.config as cfg

cfg.DB_DIR = "/home/yhe/Documents/amworkflow/usecases/param_wall/db"
dt = query_multi_data("ModelProfile")


