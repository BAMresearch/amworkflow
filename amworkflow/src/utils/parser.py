import yaml
import os
import logging
from amworkflow.src.constants.enums import Directory as D
import numpy as np
import flatdict
import ruamel.yaml
import argparse
import numpy as np

def yaml_parser(dir: str) -> dict:
    try:
        os.path.isdir(dir)
    except:
        logging.info("Wrong directory provided.")
    with open(dir, 'r') as yaml_file:
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

def cmd_parser(data, args) -> dict:
    opt = {"name": args.name,
           "geometry_parameter": {}
           }
    if args.iter_param != None:
        it_param = np.array([args.iter_param[i:i+3] for i in range(0,len(args.iter_param),3)]).T
        opt["batch_parameter"] = {"isbatch": True}
    for ind, item in enumerate(args.geom_param):
        opt["geometry_parameter"][item]=  {"startpoint": args.geom_param_value[ind],
                                           "endpoint": None,
                                           "num": None}
        if ind in it_param[0]:
            opt["geometry_parameter"][item].update({"endpoint":it_param[1][ind],"num":it_param[2][ind]})
    if args.mesh_by_layer != None:
        opt["mesh_parameter"] = {"layer_num":args.mesh_by_layer}
    if args.mesh_by_thickness != None:
        opt["mesh_parameter"] = {"layer_num":args.mesh_by_thickness}
    if args.mesh_size_factor != None:
        opt["mesh_parameter"].update({"mesh_size_factor":args.mesh_size_factor})
    if args.stl_linear_deflect != None:
        opt["stl_parameter"] = {"linear_deflection":args.stl_linear_deflect}
    if args.stl_angular_deflect != None:
        opt["stl_parameter"].update({"angular_deflection":args.stl_angular_deflect})
    return opt
    
