import yaml
import os
import logging
from amworkflow.src.constants.enums import Directory as D
import numpy as np
import flatdict
import ruamel.yaml
import argparse

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
        # return dict_flat(input_data)
        return input_data

def dict_flat(input_dict: dict) -> dict:
    return flatdict.FlatDict(input_dict)

# parser = argparse.ArgumentParser(description='amworkflow.')
# parser.add_argument("batch",
#                     metavar='b',
#                     type = bool,
#                     help='Batch mode. True for creating multiple geometries',
#                     default= True)
# parser.add_argument("usecase",
#                     )
# args = parser.parse_args()
# data_mapper(yaml_parser(D.USECASE_PATH_PARAMWALL_PATH.value, "test1.yaml"))
# print(args.batch)
# print(yaml_parser(D.USECASE_PATH_PARAMWALL_PATH.value, "test1.yaml")["batch_parameter"]["isbatch"])
