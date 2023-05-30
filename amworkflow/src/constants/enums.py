from enum import Enum
import os.path as op

class Directory(Enum):
    SYS_PATH = op.dirname(op.dirname(op.dirname(op.dirname(__file__))))
    RETURN_ONE_LAYER = "/.."
    FREECAD_PATH = r'/freecad_appimage/squashfs-root/usr/lib/'
    STL_OUTPUT_DIR = ""
    DATABASE_FILE_PATH = op.dirname(op.dirname(__file__)) + r'/infrastructure/database/files/'
    USECASE_PATH = SYS_PATH + r'/usecases'
    USECASE_PATH_PARAMWALL_PATH = USECASE_PATH + r'/param_wall/'

class Mapper(Enum):
    LENGTH = 'geometry_parameter:length:length'
    HEIGHT = 'geometry_parameter:height:length'
    WIDTH = 'geometry_parameter:width:length'
    RADIUS = 'geometry_parameter:curve:radius'
    L_ENDPOINT = 'geometry_parameter:length:endpoint'
    H_ENDPOINT = 'geometry_parameter:height:endpoint'
    W_ENDPOINT = 'geometry_parameter:width:endpoint'
    R_ENDPOINT = 'geometry_parameter:curve:endpoint'
    L_NUM = 'geometry_parameter:length:num'
    H_NUM = 'geometry_parameter:height:num'
    W_NUM = 'geometry_parameter:width:num'
    R_NUM = 'geometry_parameter:curve:num'
    LIN_DEFLECT = 'stl_parameter:linear_deflection'
    ANG_DEFLECT = 'stl_parameter:angular_deflection'
    ISBATCH = 'batch_parameter:isbatch'
    WITHCURVE = 'geometry_parameter:with_curve'
    
class Timestamp(Enum):
    YY_MM_DD_HH_MM_SS = "%y%m%d%H%M%S"
    YYYY_MM_DD_HH_MM = "%Y%m%d%H%M"
    
class InputFormat(Enum):
    BATCH_NUM = "batch_num"
    WITH_CURVE = "withCurve"
    LENGTH = "length"
    WIDTH = "width"
    HEIGHT = "height"
    RADIUS = "radius"
    LIN_DEFLECT = "linear_deflection"
    ANG_DEFLECT = "angular_deflection"   
    FILE_NAME = "filename"
    STL_HASHNAME = "stl_hashname"