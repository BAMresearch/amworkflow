import logging
import os
from enum import Enum, auto

import yaml
from pint import UnitRegistry

import amworkflow.geometry.builtinCAD as bcad
from amworkflow.utils import check as chk

ureg = UnitRegistry()


def convert_units(value, from_unit, to_unit):
    return value * ureg(from_unit).to(to_unit)


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


def create_new_config(config_name):
    directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
    print(directory)
    file_path = os.path.join(directory, config_name)
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
        self.standard = "ConcretePrinter"
        self.load_standard()
        self.coordinate_system = "absolute"
        self.gcode = []
        self.header = [
            self.Absolute,
            self.ExtruderAbsolute,
            self.set_fanspeed(0),
            self.set_temperature(0),
            self.set_tool(0),
        ]
        self.tailer = [self.ExtruderOFF, self.FanOFF, self.BedOFF, self.MotorOFF]

    def load_standard(self, std: str = None):
        directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
        config_list_no_ext = [
            os.path.splitext(file)[0] for file in os.listdir(directory)
        ]
        if std is not None:
            self.standard = std
        if self.standard not in config_list_no_ext:
            raise ValueError(f"{self.standard} does not exist.")
        config = read_config(self.standard + ".yaml")
        logging.info(f"Load config {self.standard}")
        for state in PrintState:
            if state.name in config:
                setattr(self, state.name, config[state.name])

    def command(self, line: str):
        """Write one line of command to gcode file"""
        self.gcode.append(line + "\n")
        return None

    def reset_extrusion(self):
        return f"{self.Reset} {self.LengthOfExtrude}0"

    def elevate(self, z: float, f: float = None):
        cmd = f"{self.LinearMove} {self.SetZ}{z}"
        if f is not None:
            cmd += f" {self.SetFeedRate}{f}"
        self.command(cmd)
        return cmd

    def move(self, p: bcad.Pnt, e: float = None, f: float = None):
        """Move to a point in XY plane"""
        cmd = f"{self.LinearMove} {self.SetX}{p.value[0]} {self.SetY}{p.value[1]}"
        if e is not None:
            cmd += f" {self.LengthOfExtrude}{e}"
        if f is not None:
            cmd += f" {self.SetFeedRate}{f}"
        self.command(cmd)
        return cmd

    def write_gcode(self, filename: str, gcode: str):
        for line in self.tailer:
            self.command(line)
        logging.info(f"Write gcode to {filename}")
        with open(filename, "w", encoding="utf-8") as f:
            f.write("".join(gcode))

    def set_unit(self, unit):
        if unit == "mm":
            return self.UseMM
        elif unit == "inch":
            return self.UseInch
        else:
            raise ValueError("Unit must be mm or inch")

    def set_coordinate_system(self):
        if self.coordinate_system == "absolute":
            return self.command(self.Absolute)
        elif self.coordinate_system == "relative":
            return self.command(self.Relative)
        else:
            raise ValueError("Coordinate system must be absolute or relative")

    def init_gcode(self):
        for line in self.header:
            self.command(line)

    def set_fanspeed(self, speed):
        return f"{self.FanON} S{speed}"

    def set_temperature(self, temperature):
        return f"{self.SetExtruderTemperature} S{temperature}"

    def set_tool(self, tool_number):
        return f"{self.SetTool}{tool_number}"


# gcode = GcodeCommand()
# gcode.load_standard()
# print(gcode.SetXOffset)


# create_new_config("RepRap.yaml")
# loaded_data = read_config("RepRap.yaml")
# print(loaded_data)
