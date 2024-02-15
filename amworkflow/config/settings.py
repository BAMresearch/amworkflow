import logging
import subprocess
import os
ENABLE_OCC = True
ENABLE_CONCURRENT_MODE = False
ENABLE_SQL_DATABASE = True
ENABLE_SHELF = False
STORAGE_PATH = ""


def find_git_project_root():
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
            .decode("utf-8")
            .strip()
        )
    except subprocess.CalledProcessError:
        # Not in a Git repository
        return None


ROOT_PATH = find_git_project_root()
STORAGE_PATH = ROOT_PATH + "/storage"
os.makedirs(STORAGE_PATH, exist_ok=True)

project_root = find_git_project_root()
if ENABLE_SQL_DATABASE + ENABLE_SHELF == 2:
    raise ValueError("ENABLE_SQL_DATABASE and ENABLE_SHELF cannot be both True")
LOG_LEVEL = logging.INFO
