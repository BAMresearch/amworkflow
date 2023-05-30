import yaml
import os
import logging
from src.constants.enums import Directory as D
import numpy as np
import flatdict
import ruamel.yaml

def yaml_parser(dir: str,
                filename: str) -> dict:
    try:
        os.path.isdir(dir)
    except:
        logging.info("Wrong directory provided.")
    with open(dir + "/" + filename, 'r') as yaml_file:
        yaml = ruamel.yaml.YAML(typ='safe')
        yaml.allow_duplicate_keys = True
        input_data = yaml.load(yaml_file)
        return dict_flat(input_data)

def dict_flat(input_dict: dict) -> dict:
    return flatdict.FlatDict(input_dict)

# data_mapper(yaml_parser(D.USECASE_PATH_PARAMWALL_PATH.value, "test1.yaml"))

# print(yaml_parser(D.USECASE_PATH_PARAMWALL_PATH.value, "test1.yaml")["batch_parameter"]["isbatch"])
