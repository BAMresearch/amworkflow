from doit.action import CmdAction
import os
import pkg_resources
def task_hello():
    """hello cmd """

    def create_cmd_string():
        return "echo hi"

    return {
        'actions': [CmdAction(create_cmd_string)],
        'verbosity': 2,
        }

def env_check():
    
    dists = [str(d) for d in pkg_resources.working_set]
    # Filter out distributions you don't care about and use.
    ck_occu = any("OCCUtils" in s for s in dists)
    ck_amworkflow = any("amworkflow" in s for s in dists)
    return ck_amworkflow, ck_occu

def task_install():
    cmd_list = []
    aw, ocu = env_check()
    if not ocu:
        cmd_list.append("python -m pip install /amworkflow/dependencies/OCCUtils-0.1.dev0-py3-none-any.whl")
        cmd_list.append('echo occ-utils installed successfully.')
    else:
        cmd_list.append('echo occ-utils already installed.')
    if not aw:
        cmd_list.append("python -m pip install .")
    else:
        cmd_list.append('echo amworkflow already installed.')
    return {
        'actions': cmd_list,
        'verbosity': 2
    }

def task_new_case():
    def scpt_ctr(case_name):
        cwd = os.getcwd()
        ucs = os.path.join(cwd,"usecases")
        xst_dir = (os.listdir(ucs))
        if case_name not in xst_dir:
            n_dir = os.path.join(ucs, case_name)
            os.mkdir(n_dir)
            with open(f"{n_dir}/{case_name}.py", "w") as opt:
                opt.write("from amworkflow.api import amWorkflow as aw\n@aw.engine.amworkflow()\ndef geometry_spawn(pm):\n#This is where to define your model.\n\nreturn #TopoDS_Shape")
        else:
            print(f"{case_name} already exists, please input a new name.")
    
    return {
        "actions": [scpt_ctr],
        'params': [{'name': 'case_name', 'short': 'n', 'long': 'name', "default": "new_usecase"}],
        'verbosity': 2
    }
