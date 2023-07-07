from pydantic import BaseModel, ValidationError, NegativeInt, PositiveInt, conint, conlist, constr, PositiveFloat, validator
from typing import Optional
from polyfactory.factories.pydantic_factory import ModelFactory
from amworkflow.src.utils.parser import yaml_parser
from amworkflow.src.constants.enums import Directory as D

class MapParamModel(object):
    def __init__(self, label: list, data: list = None):
        self.data = data if data != None else [None for i in range(len(label))]
        self.label = label
        self.mapping()
        self.dict = dict(zip(self.label, self.data))
    
    def mapping(self):
        for ind, lbl in enumerate(self.label):
            setattr(self, lbl, self.data[ind])
    

# lab = ["a", "b", "c"]
# # val = [2, 3, 1]
# dat = MapParamModel(lab)
# print(dat.a)