from doit.action import CmdAction
# Define actions here
def environment_init() -> bool:
    import  sys, os
    sys.path.append(os.getcwd())
    return True

def run_gui() -> str:
    return "python amworkflow/src/main2.py"
# Define tasks here
def task_start() -> None:
    return {
        "actions": [environment_init, 
                    CmdAction(run_gui)]
    }