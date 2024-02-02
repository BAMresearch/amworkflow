import csv
import logging
import os
import typing
from datetime import datetime
from pathlib import Path

import numpy as np

import amworkflow.gcode.printer_config as printer_config

typing.override = lambda x: x


class Gcode:
    """Base class with API for any gcode writer."""

    def __init__(self, *args, **kwargs) -> None:
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)

    @typing.override
    def create(self, in_file: Path, out_gcode: Path) -> None:
        """Create gcode file by given path file or geometry file

        Args:
            in_file: File path to path point file or stl file from geometry step
            out_gcode File path of output gcode file.

        Returns:

        """
        raise NotImplementedError


class GcodeFromPoints(Gcode):
    """Gcode writer from path points."""

    def __init__(
        self,
        layer_num: float = 1,
        layer_height: float = 1,
        line_width: float = 1,
        offset_from_origin: np.ndarray = np.array([0, 0]),
        unit: str = "mm",
        standard: str = "ConcretePrinter",
        coordinate_system: str = "absolute",
        nozzle_diameter: float = 0.4,
        kappa: float = 1,
        tool_number: int = 0,
        feedrate: int = 1800,
        **kwargs,
    ) -> None:
        """Gcode writer from path points.
        Args:
            layer_num: Number of layers (default 1).
            layer_height: Height of each layer (default 1).
            line_width: Width of each line (default 1).
            offset_from_origin: Offset of the points (default [0,0]).
            unit: Unit (default mm).
            standard: Standard config file (default ConcretePrinter).
            coordinate_system: Coordinate system (default absolute).
            nozzle_diameter: Nozzle diameter (default 0.4).
            kappa: Coefficient of rectifying the extrusion length (default 1).
            tool_number: Tool number (default 0).
            feedrate: Feed rate (default 1800).
        """
        self.line_width = line_width
        # Width of the line
        self.layer_num = layer_num
        # Number of layers
        self.layer_height = layer_height
        # Layer height
        self.unit = unit
        self.standard = standard
        self.load_standard()
        self.coordinate_system = coordinate_system
        self.nozzle_diameter = nozzle_diameter
        self.kappa = kappa
        # Coefficient of rectifying the extrusion length
        self.tool_number = tool_number
        # Tool number
        self.feedrate = feedrate
        # Feed rate
        self.offset_from_origin = offset_from_origin
        # Offset of the points
        self.gcode = []
        # Container of gcode
        self.points = []
        # Container of points
        self.header = [
            self.Absolute,
            self.ExtruderAbsolute,
            self.set_fanspeed(0),
            self.set_temperature(0),
            self.set_tool(0),
        ]
        # Container of header of gcode
        self.tail = [self.ExtruderOFF, self.FanOFF, self.BedOFF, self.MotorOFF]
        # Container of tail of gcode
        super().__init__(**kwargs)

    def create(self, in_file: Path, out_gcode: Path) -> None:
        """Create gcode file by given path point file

        Args:
            in_file: File path to path point file
            out_gcode File path of output gcode file.

        Returns:

        """
        self.read_points(in_file)
        self.init_gcode()
        z = 0
        for i in range(len(self.points)):
            z += self.layer_height
            coordinates = self.points
            coordinates = np.round(np.vstack((coordinates, coordinates[0])), 5)
            self.elevate(z, self.feedrate)
            self.reset_extrusion()
            E = 0
            for j, coord in enumerate(coordinates):
                if j == 0:
                    self.move(coord, 0, self.feedrate)
                else:
                    extrusion_length = self.compute_extrusion(coord, coordinates[j - 1])
                    E += extrusion_length
                    self.move(coord, np.round(E, 5), self.feedrate)
        self.write_gcode(out_gcode, self.gcode)

    def compute_extrusion(self, p0: list, p1: list):
        """Compute the extrusion length. rectify the extrusion length by the kappa factor.

        :param p0: The previous point
        :type p0: list
        :param p1: The current point
        :type p1: list
        :return: The extrusion length
        :rtype: float
        """
        self.nozzle_area = 0.25 * np.pi * self.nozzle_diameter**2
        L = np.linalg.norm(p0 - p1)
        E = np.round(L * self.line_width * self.layer_height / self.nozzle_area, 4)
        if self.kappa == 0:
            logging.warning("Kappa is zero, set to 1")
            self.kappa = 1
        return E / self.kappa

    def read_points(self, csv_file: str):
        """Read points from file

        Args:
            filepath: Path to file

        Returns:
            points: List of points
        """
        self.points = (
            np.genfromtxt(csv_file, delimiter=",", skip_header=1)
            + self.offset_from_origin
        ).tolist()

    def load_standard(self, std: str = None):
        """Load standard config file

        :param std: defaults to None
        :type std: str, optional
        :raises ValueError:
        """
        directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
        config_list_no_ext = [
            os.path.splitext(file)[0] for file in os.listdir(directory)
        ]
        if std is not None:
            self.standard = std
        if self.standard not in config_list_no_ext:
            raise ValueError(f"{self.standard} does not exist.")
        config = printer_config.read_config(self.standard + ".yaml")
        logging.info(f"Load config {self.standard}")
        for state in printer_config.PrintState:
            if state.name in config:
                setattr(self, state.name, config[state.name])

    def reset_extrusion(self):
        """Reset extrusion length

        :return: string of gcode command
        :rtype: str
        """
        cmd = f"{self.Reset} {self.LengthOfExtrude}0"
        self.gcode.append(cmd + "\n")

    def elevate(self, z: float, f: float = None):
        """Elevate to a height

        :param z: z coordinate
        :type z: float
        :param f: feed rate, defaults to None
        :type f: float, optional
        :return: string of gcode command
        :rtype: str
        """
        cmd = f"{self.LinearMove} {self.SetZ}{z}"
        if f is not None:
            cmd += f" {self.SetFeedRate}{f}"
        self.gcode.append(cmd + "\n")

    def move(self, p: list, e: float = None, f: float = None):
        """Move to a point in XY plane"""
        cmd = f"{self.LinearMove} {self.SetX}{p[0]} {self.SetY}{p[1]}"
        if e is not None:
            cmd += f" {self.LengthOfExtrude}{e}"
        if f is not None:
            cmd += f" {self.SetFeedRate}{f}"
        self.gcode.append(cmd + "\n")

    def write_gcode(self, filename: str, gcode: str):
        """Write gcode to file

        :param filename: file name
        :type filename: str
        :param gcode: gcode string
        :type gcode: str
        """
        for line in self.tail:
            self.gcode.append(line + "\n")
        logging.info(f"Write gcode to {filename}")
        with open(filename, "w", encoding="utf-8") as f:
            f.write("".join(gcode))

    def set_unit(self, unit):
        """Set unit

        :param unit: unit
        :type unit: str
        :raises ValueError: Value error
        :return: string of gcode command
        :rtype: str
        """
        if unit == "mm":
            return self.UseMM
        elif unit == "inch":
            return self.UseInch
        else:
            raise ValueError("Unit must be mm or inch")

    def set_coordinate_system(self):
        """Set coordinate system

        :raises ValueError: Value error
        :return: string of gcode command
        :rtype: str
        """
        if self.coordinate_system == "absolute":
            return self.command(self.Absolute)
        elif self.coordinate_system == "relative":
            return self.command(self.Relative)
        else:
            raise ValueError("Coordinate system must be absolute or relative")

    def init_gcode(self):
        """Initialize gcode"""
        self.comment_info()
        for line in self.header:
            self.gcode.append(line + "\n")

    def set_fanspeed(self, speed):
        """Set fan speed
        :param speed: fan speed
        :type speed: float
        :return: string of gcode command
        :rtype: str
        """
        return f"{self.FanON} S{speed}"

    def set_temperature(self, temperature):
        """Set temperature

        :param temperature: temperature
        :type temperature: float
        :return: string of gcode command
        :rtype: str
        """
        return f"{self.SetExtruderTemperature} S{temperature}"

    def set_tool(self, tool_number):
        """set tool

        :param tool_number: tool number
        :type tool_number: int
        :return: string of gcode command
        :rtype: str
        """
        return f"{self.SetTool}{tool_number}"

    def comment_info(self):
        def comment(text):
            return f"; {text}\n"

        def distance(p0, p1):
            return np.linalg.norm(np.array(p0) - np.array(p1))

        print_length = 0
        for i, pt in enumerate(self.points):
            if i == 0:
                print_length += distance(pt, self.points[-1])
            else:
                print_length += distance(pt, self.points[i - 1])
        material_consumption = (
            print_length * self.line_width * self.layer_height * self.layer_num * 1e-3
        )

        points_trans = np.array(self.points).T
        length = np.max(points_trans[0]) - np.min(points_trans[0]) + self.line_width
        width = np.max(points_trans[1]) - np.min(points_trans[1]) + self.line_width

        self.gcode.append(comment(f"Timestamp: {datetime.now()}"))
        self.gcode.append(comment(f"Length: {length}"))
        self.gcode.append(comment(f"Width: {width}"))
        self.gcode.append(comment(f"Height: {self.layer_height * self.layer_num}"))
        self.gcode.append(comment(f"Layer height: {self.layer_height}"))
        self.gcode.append(comment(f"Layer number: {self.layer_num}"))
        self.gcode.append(comment(f"Line width: {self.line_width}"))
        self.gcode.append(comment(f"Tool number: {self.tool_number}"))
        self.gcode.append(comment(f"Feed rate: {self.feedrate}"))
        self.gcode.append(comment(f"Kappa: {self.kappa}"))
        self.gcode.append(comment(f"Standard: {self.standard}"))
        self.gcode.append(comment(f"Coordinate system: {self.coordinate_system}"))
        self.gcode.append(comment(f"Unit: {self.unit}"))
        self.gcode.append(comment(f"Nozzle diameter: {self.nozzle_diameter}"))

        self.gcode.append(comment(f"Material consumption(L): {material_consumption}"))
        self.gcode.append(
            comment(
                f"Original point: ({self.offset_from_origin[0]},{self.offset_from_origin[1]})"
            )
        )
