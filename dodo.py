from doit.action import CmdAction
import amworkflow.src.core.create_geometry as cg
import amworkflow.src.constants.enums as e
from amworkflow.src.utils.db_calib import db_calib
# Define actions here
def environment_init() -> bool:
    import  sys, os
    sys.path.append(os.getcwd())
    return True

def run_gui() -> str:
    return "python amworkflow/src/main2.py"

def create_wall() -> None:
    task = cg.CreateWall(yaml_file_dir=e.Directory.USECASE_PATH_PARAMWALL_PATH.value, yaml_file_name="test1.yaml")
    task.create_wall()
    task.download()
    
# Define tasks here
def task_start() -> None:
    return {
        "actions": [environment_init, 
                    db_calib]
    }
    
def task_create_wall() -> None:
    return {"actions": [create_wall]}