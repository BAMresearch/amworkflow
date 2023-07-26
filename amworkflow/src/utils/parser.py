import yaml
import os
import logging
from amworkflow.src.constants.enums import Directory as D
import numpy as np
import flatdict
import ruamel.yaml
import argparse
import numpy as np
from amworkflow.src.constants.enums import Label as L

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
# print(yaml_parser(D.USECASE_PATH_PARAMWALL_PATH.value, "test1.yaml")[L.BATCH_PARAM.value][L.IS_BATCH.value])
def cmd_parser(args) -> dict:
    opt = {}
    if args.iter_param != None:
        it_param = np.array([args.iter_param[i:i+3] for i in range(0,len(args.iter_param),3)]).T
        print(args.geom_param)
        opt[L.BATCH_PARAM.value] = {L.IS_BATCH.value: True}
    else:
        opt[L.BATCH_PARAM.value] = {L.IS_BATCH.value: False}
    if args.geom_param != None:
        opt[L.GEOM_PARAM.value] = {}
        for ind, item in enumerate(args.geom_param):
            opt[L.GEOM_PARAM.value][item]=  {L.STARTPOINT.value: args.geom_param_value[ind],
                                            L.ENDPOINT.value: None,
                                            L.NUM.value: None}
            if args.iter_param != None:
                if ind + 1 in it_param[0]:
                    opt[L.GEOM_PARAM.value][item].update({L.ENDPOINT.value:it_param[1][ind],L.NUM.value:it_param[2][ind]})
    return opt
    
