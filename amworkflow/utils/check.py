from pathlib import Path
def is_file_exist(file_path: str) -> bool:
    return Path(file_path).is_file()

def is_dir_exist(dir_path: str) -> bool:
    return Path(dir_path).is_dir()  

