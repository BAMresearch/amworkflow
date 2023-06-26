from pydantic import BaseModel, ValidationError, NegativeInt, PositiveInt, conint, conlist, constr, PositiveFloat, validator
from typing import Optional
from polyfactory.factories.pydantic_factory import ModelFactory
from src.utils.parser import yaml_parser
from src.constants.enums import Directory as D
class BatchParameter(BaseModel):
    isbatch: bool

class BaseGeometry(BaseModel):
    radius: Optional[float] = None
    length: Optional[float] = None
    endpoint: float
    num: int
    
    @validator("*")
    def validate_length_and_radius(cls, value, values):
        if 'length' in values and 'radius' in values:
            if values['length'] is not None and values['radius'] is not None:
                raise ValueError("Only one of length or radius should have a value, or both should be None.")
            elif values['length'] is None and values['radius'] is None:
                raise ValueError("Either length or radius should have a value, but not both should be None.")
            
            # elif values['length'] is not None:
            #     if values['length'] > values['endpoint']:
            #         raise ValueError("length should not be larger than endpoint.")
                
            # elif values['radius'] is not None:
            #     if values['radius'] > values['endpoint']:
            #         raise ValueError("radius should not be larger than endpoint.")  
        return value

class BaseMeshLayer(BaseModel):
    config: bool
    num: float
    
class WallGeometryParameter(BaseModel):
    length: BaseGeometry
    height: BaseGeometry
    width: BaseGeometry
    radius: BaseGeometry

class STLParameter(BaseModel):
    linear_deflection: float
    angular_deflection: float

class MeshParameter(BaseModel):
    layer_num: BaseMeshLayer
    layer_thickness: BaseMeshLayer
    mesh_size_factor: PositiveFloat
    

class WallParam(BaseModel):
    batch_parameter: BatchParameter
    mesh_parameter: MeshParameter
    geometry_parameter: WallGeometryParameter
    stl_parameter: STLParameter

class DB_WallGeometryFile(BaseModel):
    batch_num: Optional[PositiveInt]
    withCurve: Optional[bool]
    length: Optional[PositiveFloat]
    width: Optional[PositiveFloat]
    height: Optional[PositiveFloat]
    radius: Optional[float]
    linear_deflection: Optional[PositiveFloat]
    angular_deflection: Optional[PositiveFloat]
    filename: Optional[str]
    stl_hashname: Optional[constr(max_length=32, min_length=32)]
    
class DB_XdmfFile(BaseModel):
    xdmf_hashname: Optional[constr(max_length=32, min_length=32)]
    mesh_size_factor: Optional[PositiveFloat]
    layer_thickness: Optional[ PositiveFloat]
    layer_num: Optional[PositiveInt]
    batch_num: Optional[PositiveInt]
    stl_hashname: Optional[constr(max_length=32, min_length=32)]
    xdmf_name: Optional[str]

class DB_H5File(BaseModel):
    h5_hashname: Optional[constr(max_length=32, min_length=32)]
    batch_num: Optional[PositiveInt]
    xdmf_hashname: Optional[constr(max_length=32, min_length=32)]
    h5_name: Optional[str]
    
class DBFactory(ModelFactory[DB_WallGeometryFile]):
    __model__ = DB_WallGeometryFile

# data = yaml_parser(D.USECASE_PATH_PARAMWALL_PATH.value, "test1.yaml")
# # print(data)
# parseData = WallParam(**data)
# for key, item in parseData.geometry_parameter:
#     if key == "with_curve":
#         print(key, item)
#         continue
#     # try: print(key, item.length, item.endpoint)
#     # except: print(key, item.radius, item.endpoint)
#     print(key, item.radius, item.length)

# db_model = DBFactory.build()
# print(db_model.dict())