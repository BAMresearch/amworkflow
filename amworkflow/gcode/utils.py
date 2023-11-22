import os
from enum import Enum, auto

import yaml
from pint import UnitRegistry

import amworkflow.geometry.builtinCAD as bcad
from amworkflow.utils import check as chk

ureg = UnitRegistry()

distance = 5 * ureg.millimeter
distance.ito(ureg.inch)
print(distance)


def convert_units(value, from_unit, to_unit):
    return value * ureg(from_unit).to(to_unit)


class PrintProcess:
    def __init__(self):
        self.unit = "mm"


# print(convert_units(5, "mm", "inch"))


class PrintState(Enum):
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
    MotorOn = auto()
    MotorOff = auto()
    FanON = auto()
    FanOff = auto()
    ExtruderONForward = auto()
    ExtruderONReverse = auto()
    ExtruderOFF = auto()
    BedON = auto()
    BedOFF = auto()
    SetBedTempature = auto()
    SetExtruderTempature = auto()


def create_new_config(config_name):
    directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
    print(directory)
    file_path = os.path.join(directory, config_name)
    print(file_path)
    print(chk.is_file_exist(file_path))
    if chk.is_file_exist(file_path):
        raise FileExistsError(f"{config_name} already exists.")
    enum_members_list = [state.name for state in PrintState]
    # Write data to the YAML file
    with open(file_path, "w", encoding="utf-8") as yaml_file:
        yaml.dump(enum_members_list, yaml_file, default_flow_style=False)


def read_config(config_name):
    directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
    file_path = os.path.join(directory, config_name)
    if not chk.is_file_exist(file_path):
        raise FileNotFoundError(f"{config_name} does not exist.")
    with open(file_path, "r", encoding="utf-8") as yaml_file:
        loaded_data = yaml.safe_load(yaml_file)
        flattened_data = {k: v for d in loaded_data for k, v in d.items()}
        return flattened_data


class GcodeCommand:
    def __init__(self) -> None:
        self.unit = "mm"
        self.standard = "RepRap"
        self.coordinate_system = "absolute"

    def load_standard(self):
        directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
        config_list_no_ext = [
            os.path.splitext(file)[0] for file in os.listdir(directory)
        ]
        if self.standard not in config_list_no_ext:
            raise ValueError(f"{self.standard} does not exist.")
        config = read_config(self.standard + ".yaml")
        for state in PrintState:
            if state.name in config:
                setattr(self, state.name, config[state.name])

    def command(self, line: str):
        """Write one line of command to gcode file"""
        return line + "\n"

    def move(self, p: bcad.Pnt, e: float = None):
        """Move to a point in space"""
        if e is None:
            return self.command(f"G1 X{p.value[0]} Y{p.value[1]}")
        return self.command(f"G1 X{p.value[0]} Y{p.value[0]} E{e}")

    def write_gcode(self, filename: str, gcode: str):
        with open(filename, "w", encoding="utf-8") as f:
            f.write(gcode)

    def set_unit(self, unit):
        if unit == "mm":
            return self.command("G21")
        elif unit == "inch":
            return self.command("G20")
        else:
            raise ValueError("Unit must be mm or inch")

    def set_coordinate_system(self):
        if self.coordinate_system == "absolute":
            return self.command("G90")
        elif self.coordinate_system == "relative":
            return self.command("G91")
        else:
            raise ValueError("Coordinate system must be absolute or relative")


gcode = GcodeCommand()
gcode.load_standard()
print(gcode.SetXOffset)


# create_new_config("RepRap.yaml")
# loaded_data = read_config("RepRap.yaml")
# print(loaded_data)
