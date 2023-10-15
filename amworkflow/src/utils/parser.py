import os
import logging
from amworkflow.src.constants.enums import Directory as D
import numpy as np
import flatdict
import ruamel.yaml
import argparse
import numpy as np
from amworkflow.src.constants.enums import Label as L
from amworkflow.src.constants.exceptions import InsufficientDataException
from amworkflow.src.constants.data_model import DeepMapParamModel

def yaml_parser(dir: str) -> dict:
    try:
        os.path.isdir(dir)
    except:
        logging.info("Wrong directory provided.")
    with open(dir, 'r') as yaml_file:
        yaml = ruamel.yaml.YAML(typ='safe')
        yaml.allow_duplicate_keys = True
        ipd = yaml.load(yaml_file)
        # return dict_flat(input_data)
        ipd = yaml_translater(ipd)
        return ipd

def dict_flat(input_dict: dict) -> dict:
    return flatdict.FlatDict(input_dict)

def geom_param_parser(args) -> dict:
    opt = {L.BATCH_PARAM.value: {L.IS_BATCH.value: True}}
    if args.iter_param != None:
        opt[L.BATCH_PARAM.value] = {L.IS_BATCH.value: True}
        it_param = np.array([args.iter_param[i:i+3] for i in range(0,len(args.iter_param),3)]).T
    if args.geom_param != None:
        opt[L.GEOM_PARAM.value] = {}
        for ind, item in enumerate(args.geom_param):
            opt[L.GEOM_PARAM.value][item]=  {L.STARTPOINT.value: args.geom_param_value[ind],
                                            L.ENDPOINT.value: None,
                                            L.NUM.value: None}
            if args.iter_param != None:
                if ind + 1 in it_param[0]:
                    ind = np.where(it_param[0] == ind + 1)[0][0]
                    opt[L.GEOM_PARAM.value][item].update({L.ENDPOINT.value:it_param[1][ind],L.NUM.value:it_param[2][ind]})
        return opt
    
def yaml_translater(raw: dict):
    opt = DeepMapParamModel({})
    opt.name = raw[L.MDL_PROF.value][L.MDL_NAME.value]
    opt.import_dir = raw[L.MDL_PROF.value][L.IMP_DIR.value]
    opt.geom_param = None
    opt.geom_param_value = None
    opt.iter_param = None
    opt.mesh_by_layer = raw[L.MESH_PARAM.value][L.LYR_NUM.value]
    opt.mesh_by_thickness = raw[L.MESH_PARAM.value][L.LYR_TKN.value]
    opt.stl_angular_deflect = raw[L.STL_PARAM.value][L.ANG_DFT.value]
    opt.stl_linear_deflect = raw[L.STL_PARAM.value][L.LNR_DFT.value]
    geom = raw[L.GEOM_PARAM.value]
    gp = []
    gpv = []
    ip = []
    i = 0
    for k,v in geom.items():
        gp.append(k)
        i += 1
        for kk in v.keys():
            if kk == L.STARTPOINT.value:
                gpv.append(v[kk])
            if (kk == L.ENDPOINT.value) and (v[kk] is not None):
                ip.append(i)
                ip.append(v[kk])
                ip.append(v[L.NUM.value])
    if len(gp) != 0:
        opt.geom_param = gp
    if len(gpv) != 0:
        opt.geom_param_value = gpv
    if len(ip) != 0:
        opt.iter_param = ip
    return opt
        
def batch_data_parser(param_num: int, iter_param: list) -> np.ndarray:
    it_param = np.array([iter_param[i:i+3] for i in range(0,len(iter_param),3)]).T
    end_vec = []
    num_vec = []
    for ind in range(param_num):
        if ind + 1 in it_param[0]:
            ind = np.where(it_param[0] == ind + 1)[0][0]
            end_vec.append(it_param[1][ind])
            num_vec.append(it_param[2][ind])
        else:
            end_vec.append(None)
            num_vec.append(None)
    return np.array(end_vec), np.array(num_vec)
    