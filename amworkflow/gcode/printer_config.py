import os
from enum import Enum, auto

import yaml


class PrintState(Enum):
    """Print state. This is used to define the state of the printer. Since one state can be assigned to multiple G-code commands, the state is defined as an Enum.
    :param Enum:
    :type Enum: class
    """

    UseMM = 1
    UseInch = 2
    LinearMove = auto()
    RapidMove = auto()
    ArcMove = auto()
    Reset = auto()
    Absolute = auto()
    Relative = auto()
    SetFeedRate = auto()
    SetSpindleSpeed = auto()
    SetTool = auto()
    SetX = auto()
    SetY = auto()
    SetZ = auto()
    SetXOffset = auto()
    SetYOffset = auto()
    SetZOffset = auto()
    LengthOfExtrude = auto()
    SetExtrudeSpeed = auto()
    CommandParameter = auto()
    MotorON = auto()
    MotorOFF = auto()
    FanON = auto()
    FanOFF = auto()
    ExtruderONForward = auto()
    ExtruderONReverse = auto()
    ExtruderAbsolute = auto()
    ExtruderOFF = auto()
    BedON = auto()
    BedOFF = auto()
    SetBedTemperature = auto()
    SetExtruderTemperature = auto()
    SpindleOn = auto()
    SpindleOff = auto()
    Pause = auto()
    # inserted for BAM powder bed printer
    PrinterName = auto()
    NozzleNum = auto()                
    NozzleOpen = auto()               
    PrinterX = auto()                 
    PrinterY = auto()                 
    PrinterZ = auto()                 
    VoxelDimX = auto()                
    VoxelDimY = auto()                
    VoxelDimZ = auto()                
    PrintSpeedX = auto()              
    PrintSpeedY = auto()              
    PrintSpeedZ = auto()              
    LayingSpeedX = auto()             
    ManualSpeedX = auto()             
    ManualSpeedY = auto()             
    ManualSpeedZ = auto()             
    LinesNum = auto()                        
    LayerNumMax = auto()                     
    DeltaExtraPlaneEnd = auto()              
    DeltaExtraPlaneStart = auto()            
    VoxelRaiseBeforeLaying = auto()                
    VoxelRaiseBeforePrinting = auto()        
    RecoaterOpeningPositionLaying = auto()   
    RecoaterClosingPositionLaying = auto()   
    RecoaterOpeningPositionPrinting = auto() 
    RecoaterClosingPositionPrinting = auto() 
    RecoaterOpenCloseSpeed = auto()          
    RecoaterHoleOpening = auto()             
    LumpBreakerPowder = auto()               
    MinLiquid1LevelRange1To7 = auto()        
    MaxLiquid1LevelRange1To7 = auto()        
    MinLiquid2LevelRange1To7 = auto()        
    MaxLiquid2LevelRange1To7 = auto()        
    
class CommentInfo(Enum):
    TIMESTAMP = "Timestamp"
    LENGTH = "Length"
    WIDTH = "Width"
    HEIGHT = "Height"
    LAYER_HEIGHT = "Layer Height"
    LAYER_NUMBER = "Layer Number"
    LINE_WIDTH = "Line Width"
    DIFFERENT_GEOMETRY_PER_LAYER = "Different Geometry Per Layer"
    TOOL_NUMBER = "Tool Number"
    FEED_RATE = "Feed Rate"
    KAPPA = "Kappa"
    GAMMA = "Gamma"
    DELTA = "Delta"
    STANDARD = "Standard"
    COORDINATE_SYSTEM = "Coordinate System"
    NOZZLE_DIAMETER = "Nozzle Diameter"
    UNIT = "Unit"
    MATERIAL_CONSUMPTION = "Material Consumption"
    ESTIMATED_TIME_CONSUMPTION = "Estimated Time consumption"
    ORIGINAL_POINT = "Original Point"


def create_new_config(path_config_file):
    """Create new config file

    :param path_config_file: file path of config file
    :type config_name: str
    :raises FileExistsError: File already exists
    """

    if os.path.exists(path_config_file):
        raise FileExistsError(f"{path_config_file} already exists.")
    enum_members_list = [state.name for state in PrintState]
    # Write data to the YAML file
    with open(path_config_file, "w", encoding="utf-8") as yaml_file:
        yaml.dump(enum_members_list, yaml_file, default_flow_style=False)


def read_config(path_config_file):
    """Read config file

    :param path_config_file: file path of config file
    :type config_name: str
    :raises FileNotFoundError: File not found
    :return: config data
    :rtype: dict
    """
    # directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
    # file_path = os.path.join(directory, config_name)
    if not os.path.exists(path_config_file):
        raise FileNotFoundError(f"{path_config_file} does not exist.")
    with open(path_config_file, "r", encoding="utf-8") as yaml_file:
        loaded_data = yaml.safe_load(yaml_file)
        flattened_data = {k: v for d in loaded_data for k, v in d.items()}
        return flattened_data
