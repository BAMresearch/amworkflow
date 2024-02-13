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


def create_new_config(config_name):
    """Create new config file

    :param config_name: config file name
    :type config_name: str
    :raises FileExistsError: File already exists
    """
    directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
    print(directory)
    file_path = os.path.join(directory, config_name)
    if os.path.exists(file_path):
        raise FileExistsError(f"{config_name} already exists.")
    enum_members_list = [state.name for state in PrintState]
    # Write data to the YAML file
    with open(file_path, "w", encoding="utf-8") as yaml_file:
        yaml.dump(enum_members_list, yaml_file, default_flow_style=False)


def read_config(config_name):
    """Read config file

    :param config_name: config file name
    :type config_name: str
    :raises FileNotFoundError: File not found
    :return: config data
    :rtype: dict
    """
    directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
    file_path = os.path.join(directory, config_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{config_name} does not exist.")
    with open(file_path, "r", encoding="utf-8") as yaml_file:
        loaded_data = yaml.safe_load(yaml_file)
        flattened_data = {k: v for d in loaded_data for k, v in d.items()}
        return flattened_data
