from doit.action import CmdAction
import os
import shutil
from doit import get_var
import pkg_resources


## global parameter for tasks
param = {
    "case_name": get_var('case_name', 'new_case'),
}

def env_check():
    dists = [str(d) for d in pkg_resources.working_set]
    # Filter out distributions you don't care about and use.
    ck_occu = any("OCCUtils" in s for s in dists)
    ck_amworkflow = any("amworkflow" in s for s in dists)
    return ck_amworkflow, ck_occu

def scpt_ctr(case_name):
    cwd = os.getcwd()
    ucs = os.path.join(cwd, "usecases")
    xst_dir = (os.listdir(ucs))
    if case_name not in xst_dir:
        n_dir = os.path.join(ucs, case_name)
        os.mkdir(n_dir)
        print(n_dir)
        print(os.path.join(cwd, "template.py"))
        print(os.path.join(n_dir, f"dodo_{case_name}.py"))

        shutil.copy(os.path.join(cwd, "template.py"), os.path.join(n_dir, f"{case_name}.py"))

    else:
        print(f"{case_name} already exists, please input a new name.")

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

    return {
        "actions": [(scpt_ctr,[param["case_name"]])],
        'verbosity': 2
    }