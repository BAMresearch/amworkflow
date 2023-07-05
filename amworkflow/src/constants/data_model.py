from pydantic import BaseModel, ValidationError, NegativeInt, PositiveInt, conint, conlist, constr, PositiveFloat, validator
from typing import Optional
from polyfactory.factories.pydantic_factory import ModelFactory
from amworkflow.src.utils.parser import yaml_parser
from amworkflow.src.constants.enums import Directory as D

class MapParamModel(object):
    def __init__(self, data: list, label: list):
        self.data = data
        self.label = label
        self.mapping()
    
    def mapping(self):
        for ind, lbl in enumerate(self.label):
            setattr(self, lbl, self.data[ind])

# lab = ["a", "b", "c"]
# val = [2, 3, 1]
# dat = MapParamModel(val, lab)
# print(dat.a)