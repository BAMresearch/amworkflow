import csv
import logging
import sys
import os
import re
import typing
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

import amworkflow.gcode.printer_config as printer_config
from amworkflow.geometry import builtinCAD as bcad

import stltovoxel
#import pyvista as pv

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
    
class PowderbedCodeFromSTL(Gcode):
    """Print instructions writer from stl file"""

    def __init__(
        self,
        standard: str = "PowderBedBAM",
        in_file_path: str = None,
        **kwargs,
    ) -> None:
        # Unit of the geometry
        self.standard = standard
        # Standard of the printer firmware
        # Careful, the file "printer_config.py" also has to be changed whenever parameters in the config are added/removed
        self.load_standard()
        # Path to the input file
        self.in_file_path = in_file_path
                        
        super().__init__(**kwargs)

    def create(self, in_file: Path, out_dsmn: str, out_xyz: str, out_dsmn_dir: Path = None) -> None:
        """Create dsmn printer instructions file by given stl file

        Args:
            in_file: File path to path point file
            out_dsmn: Name of output dsmn file.
            out_xyz: Name of output xyz file.
            out_dsmn_dir: Directory of output dsmn file. If not given, One output folder will be created in the root directory of the project.

        Returns:

        """

        ### Recalculate parameters
        self.RecoaterClosingPositionLaying = np.floor(self.PrinterX*1000) + 100
        self.RecoaterClosingPositionPrinting = np.floor(self.PrinterX*1000) + 200
        self.VoxelDimY = self.PrinterY / self.NozzleNum
        self.VoxelDimX = self.VoxelDimY
        self.VoxelDimZ = self.VoxelDimY

        ### creating of paths for various stuff
        stl_name, stl_fileending = os.path.splitext(os.path.basename(in_file))
        #working_folder = os.getcwd()

        recipe_name = "DSMR-" + stl_name + stl_fileending

        if out_dsmn_dir is None:
            current_directory = os.getcwd()
            root_directory = os.path.dirname(current_directory)
            out_dsmn_dir = os.path.join(current_directory, "output")


            if not os.path.exists(out_dsmn_dir):
                os.makedirs(out_dsmn_dir)
        out_dsmn_dir = Path(out_dsmn_dir)
        out_log = f"log_{stl_name}.txt"
        log_file_path = out_dsmn_dir / out_log

        ####
        ### Convert to xyz file (one file total)
        out = stltovoxel.convert_file(in_file, out_xyz, voxel_size = self.VoxelDimY*self.STLUnit, parallel = False)

        ### Load xyz file for further processing
        # Read xyz file and parse coordinates
        with open(out_xyz, 'r') as file:
            lines = file.readlines()

        # Extract x, y, z coordinates from each line
        coordinates = [list(map(float, line.strip().split())) for line in lines]

        # Convert coordinates to a NumPy array
        voxel_array = np.array(coordinates) / self.STLUnit

        # Find unique values and sort ascending (1st col: x, 2nd col: y, 3rd col: z)
        unique_x = np.unique(voxel_array[:,0])
        unique_y = np.unique(voxel_array[:,1])
        unique_z = np.unique(voxel_array[:,2])

        # Get real voxel dimension produced by stl2voxel
        sorted_x_temp = np.sort(unique_x)
        sliced_voxel_dim_x = (sorted_x_temp[1]-sorted_x_temp[0])
        sorted_y_temp = np.sort(unique_y)
        sliced_voxel_dim_y = (sorted_y_temp[1]-sorted_y_temp[0])
        sorted_z_temp = np.sort(unique_z)
        sliced_voxel_dim_z = (sorted_z_temp[1]-sorted_z_temp[0])

        # Create arrays for dimensions, adding small offset to stop-value, because it may not be included
        sorted_x = np.arange(np.min(unique_x), np.max(unique_x)+0.01*self.VoxelDimX, sliced_voxel_dim_x)
        sorted_y = np.arange(np.min(unique_y), np.max(unique_y)+0.01*self.VoxelDimY, sliced_voxel_dim_y)
        sorted_z = np.arange(np.min(unique_z), np.max(unique_z)+0.01*self.VoxelDimZ, sliced_voxel_dim_z)

        # Prepare array
        voxel_3d_array = np.zeros((sorted_x.shape[0], sorted_y.shape[0], sorted_z.shape[0]))

        # Sort ones where a coordinate exists into 3d array
        for i, row in enumerate(voxel_array):

            x_index = np.argmin(np.abs(sorted_x - row[0]))
            y_index = np.argmin(np.abs(sorted_y - row[1]))
            z_index = np.argmin(np.abs(sorted_z - row[2]))

            voxel_3d_array[x_index, y_index, z_index] = 1

        # Calculate optimized printer_x and lines_num for less printing duration, also recoater values
        self.PrinterX = (voxel_3d_array.shape[1] + 2*self.AddZeros) * self.VoxelDimX
        self.LinesNum = voxel_3d_array.shape[1] + 2*self.AddZeros
        self.RecoaterClosingPositionLaying = np.floor(self.PrinterX*1000) + 100
        self.RecoaterClosingPositionPrinting = np.floor(self.PrinterX*1000) + 200

        ### Write header - printer parameters
        with open(out_dsmn, "w") as file:
            file.write(recipe_name + "\n")
            file.write(self.PrinterName + "\n")
            file.write(str(self.VoxelDimY*1000) + "\n")
            file.write(str(self.PrinterY*1000) + "\n")
            file.write(str(self.VoxelDimX*1000) + "\n")
            file.write(str(np.round(self.PrinterX*1000, 1)) + "\n")
            file.write(str(self.NozzleOpen) + "\n")
            file.write(str(self.NozzleNum) + "\n")
            file.write(str(self.VoxelDimZ*1000) + "\n")
            file.write(str(self.PrinterZ*1000) + "\n")
            file.write(str(self.PrintSpeedX) + "\n")
            file.write(str(self.PrintSpeedY) + "\n")
            file.write(str(self.PrintSpeedZ) + "\n")
            file.write(str(self.LayingSpeedX) + "\n")
            file.write(str(self.ManualSpeedX) + "\n")
            file.write(str(self.ManualSpeedY) + "\n")
            file.write(str(self.ManualSpeedZ) + "\n")
            file.write(str(self.LinesNum) + "\n")
            file.write(str(self.LayerNumMax) + "\n")
            file.write(str(self.DeltaExtraPlaneEnd) + "\n")
            file.write(str(self.VoxelRaiseBeforeLaying) + "\n")
            file.write(str(self.VoxelRaiseBeforePrinting) + "\n")
            file.write(str(self.DeltaExtraPlaneStart) + "\n")
            file.write(str(self.RecoaterOpeningPositionLaying) + "\n")
            file.write(str(self.RecoaterClosingPositionLaying) + "\n")
            file.write(str(self.RecoaterOpeningPositionPrinting) + "\n")
            file.write(str(self.RecoaterClosingPositionPrinting) + "\n")
            file.write(str(self.RecoaterOpenCloseSpeed) + "\n")
            file.write(str(self.RecoaterHoleOpening) + "\n")
            file.write(str(self.LumpBreakerPowder) + "\n")
            file.write(str(self.MinLiquid1LevelRange1To7) + "\n")
            file.write(str(self.MaxLiquid1LevelRange1To7) + "\n")
            file.write(str(self.MinLiquid2LevelRange1To7) + "\n")
            file.write(str(self.MaxLiquid2LevelRange1To7) + "\n")
            file.write("\n"*67)   # Values from 34 to 100 are not existant, therefore empty lines

        ### Write body - voxel to hex
        # Check if there are more voxels along x than nozzles
        if voxel_3d_array.shape[0] > self.NozzleNum:
            raise ValueError(f"{voxel_3d_array.shape[0]} x-voxels to only {self.NozzleNum} nozzles, rescale model to fit within printer width of {self.PrinterY}m")
        # Check if 4 bit to hex encoding possible
        if self.NozzleNum % 4 != 0:
            raise ValueError("Number of nozzles not a multiple of 4, 4bit to hex encoding not possible")
        
        # Prepare array
        printerbed_3d_array = np.zeros((self.NozzleNum, self.LinesNum, sorted_z.shape[0]))

        # Sort into temporary array
        for ii in range(voxel_3d_array.shape[2]):
            # Position smaller prints into larger printer bed, roughly in the middle
            A_large = printerbed_3d_array[:,:,ii]
            A_small = voxel_3d_array[:,:,ii]
            start_row = (A_large.shape[0] - A_small.shape[0]) // 2
            start_col = (A_large.shape[1] - A_small.shape[1]) // 2

            A_large[start_row:start_row + A_small.shape[0], start_col:start_col + A_small.shape[1]] = A_small
            printerbed_3d_array[:,:,ii] = A_large

        # Print the actual dsmn-file
        for layers in range(printerbed_3d_array.shape[2]):
            printer_hex = []
            temp_array = np.transpose(printerbed_3d_array[:,:,layers])

            for row in temp_array:
                # Partitioning of rows in parts of length 4
                chunks = [row[i:i+4] for i in range(0, len(row), 4)]

                # Convert every chunk into binary number and then into hex number
                hex_values = [hex(int(''.join(map(str, chunk.astype(int))), 2))[2:].upper() for chunk in chunks]

                # Combine hex numbers
                printer_hex.append(hex_values)

            with open(out_dsmn, "a") as file:
                for hex_values in printer_hex:
                    file.write("".join(hex_values) + "\n")
                if self.DebugMode:
                    file.write("-" * (int(self.NozzleNum/4)) + "\n") # line separator between layers

        # # Visualize voxels with pyvista
        # def visualize_voxels(voxel_centers):
        
        #     # Create a point cloud representation
        #     point_cloud = pv.PolyData(voxel_centers) #pv.PointCloud(voxel_polydata)

        #     # Plot the point cloud using pyvista plotter
        #     point_cloud.plot(eye_dome_lighting=True)
        # visualize_voxels(voxel_array)

        ### Print some data for the user to file
        orig_stdout = sys.stdout
        f = open(log_file_path, "w")
        sys.stdout = f
        print("-- Printing and slicing statistics --")
        # Number of voxels
        tot_voxel_num = np.count_nonzero(voxel_3d_array)
        print(f"Number of voxels: {tot_voxel_num}")
        # Total voxel volume
        tot_voxel_volume = tot_voxel_num * self.VoxelDimX * self.VoxelDimY * self.VoxelDimZ
        print(f"Print volume: {tot_voxel_volume*1000:.3f} liters")
        # Number of layers
        print(f"Number of layers: {unique_z.shape[0]}")
        # Requested voxel dimensions
        print("Requested voxel dimensions:")
        print(f"\t x: {self.VoxelDimX*1000:.3f} mm")
        print(f"\t y: {self.VoxelDimY*1000:.3f} mm")
        print(f"\t z: {self.VoxelDimZ*1000:.3f} mm")
        # Actual voxel dimensions
        warning_x = ""
        warning_y = ""
        warning_z = ""
        target_deviation = 0.05
        if np.abs(1 - sliced_voxel_dim_x/self.VoxelDimX) > target_deviation:
            warning_x = " \t -> Voxel size deviation threshold reached"
        if np.abs(1 - sliced_voxel_dim_y/self.VoxelDimY) > target_deviation:
            warning_y = " \t -> Voxel size deviation threshold reached"
        if np.abs(1 - sliced_voxel_dim_z/self.VoxelDimZ) > target_deviation:
            warning_z = " \t -> Voxel size deviation threshold reached"
        print("Actual voxel dimensions:")
        print(f"\t x: {sliced_voxel_dim_x*1000:.3f} mm" + warning_x)
        print(f"\t y: {sliced_voxel_dim_y*1000:.3f} mm" + warning_y)
        print(f"\t z: {sliced_voxel_dim_z*1000:.3f} mm" + warning_z)
        sys.stdout = orig_stdout
        f.close()

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
        gamma: float = -1,
        delta: float = 1,
        tool_number: int = 0,
        feedrate: int = 1800,
        fixed_feedrate: bool = False,
        rotate: bool = False,
        density: float = 1,
        in_file_path: str = None,
        **kwargs,
    ) -> None:
        self.line_width = line_width
        # Width of the line
        self.layer_num = layer_num
        # Number of layers
        self.layer_height = layer_height
        # Layer height
        self.unit = unit
        # Unit of the geometry
        self.standard = standard
        # Standard of the printer firmware
        self.load_standard()
        self.coordinate_system = coordinate_system
        # Coordinate system of the printer firmware
        self.nozzle_diameter = nozzle_diameter
        # Diameter of the nozzle
        self.nozzle_area = 0.25 * np.pi * self.nozzle_diameter**2
        # Area of the nozzle
        self.kappa = kappa
        # Coefficient of rectifying the extrusion length
        self.gamma = gamma
        # Coefficient of rectifying the feedrate, as well as the line width
        self.delta = delta
        # Coefficient of rectifying the feedrate, as well as the line width
        self.tool_number = tool_number
        # Tool number
        self.feedrate = feedrate
        # Feed rate
        self.density = density
        # Density of the material
        self.fixed_feedrate = fixed_feedrate
        # switch of fixed feedrate
        self.offset_from_origin = offset_from_origin
        # Offset of the points
        self.print_length = 0
        # Length of the print
        self.material_consumption = 0
        # Material consumption
        self.time_consumption = 0
        # Time consumption
        self.timedelta = 0
        # Time delta
        self.points_t = []
        # Transposed points
        self.bbox = []
        # Bounding box of the points
        self.bbox_length = 0
        # Length of the bounding box
        self.bbox_width = 0
        # Width of the bounding box
        self.gcode = []
        # Container of gcode
        self.points = []
        # Container of points
        self.header = [
            self.Absolute,
            self.ExtruderAbsolute,
            self.set_fanspeed(0),
            self.set_tool(0),
        ]
        # Container of header of gcode
        self.tail = [self.ExtruderOFF, self.FanOFF, self.MotorOFF]
        # Container of tail of gcode
        self.rotate = rotate
        # rotate the model in 90 degree
        self.extrusion_tracker = []
        # Container of extrusion logs
        self.in_file_path = in_file_path
        # Path to the input file
        self.diff_geo_per_layer = False
        # switch of different geometry per layer
        
        super().__init__(**kwargs)

    def create(self, in_file: Path, out_gcode: str, out_gcode_dir: Path = None) -> None:
        """Create gcode file by given path point file

        Args:
            in_file: File path to path point file
            out_gcode: Name of output gcode file.
            out_gcode_dir: Directory of output gcode file. If not given, One output folder will be created in the root directory of the project.

        Returns:

        """
        self.read_points(in_file)

        if out_gcode_dir is None:
            current_directory = os.getcwd()
            root_directory = os.path.dirname(current_directory)
            out_gcode_dir = os.path.join(root_directory, "output_gcode")
            if not os.path.exists(out_gcode_dir):
                os.makedirs(out_gcode_dir)
        out_gcode_dir = Path(out_gcode_dir)
        self.init_gcode()
        z = 0
        for i in range(self.layer_num):
            z += self.layer_height
            self.elevate(z)
            self.reset_extrusion()
            coordinates = self.points
            coordinates = np.round(np.vstack((coordinates, coordinates[0])), 5)
            E = 0
            for j, coord in enumerate(coordinates):
                if i == 0 and j == 0:
                    extrusion_length = self.compute_extrusion(
                        coord, np.zeros_like(coord)
                    )
                else:
                    extrusion_length = (
                        self.compute_extrusion(coord, coordinates[j - 1])
                        if j > 0
                        else 0
                    )
                E += extrusion_length
                self.move(coord, np.round(E, 5), self.feedrate)
        gcode_file_path = out_gcode_dir / out_gcode
        self.write_gcode(gcode_file_path, self.gcode)
        out_log = f"log_{out_gcode}.csv"
        log_file_path = out_gcode_dir / out_log
        self.write_log(log_file_path)

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
        rect_E = E / self.kappa
        self.log_consumption(L, self.feedrate, E)
        return rect_E

    def log_consumption(self, dist, speed, material_consumption):
        time_consumption = dist / (speed / 60)
        volume = material_consumption * self.nozzle_area * 1e-6  # in L
        if len(self.extrusion_tracker) == 0:
            aggregate_volume = volume
            aggregate_time = time_consumption
        else:
            aggregate_volume = self.extrusion_tracker[-1][2] + volume
            aggregate_time = self.extrusion_tracker[-1][0] + time_consumption
        mass = aggregate_volume * self.density  # in g
        self.extrusion_tracker.append(
            [aggregate_time, volume, aggregate_volume, mass, speed]
        )

    def write_log(self, filename: str):
        with open(filename, "w") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["time", "volume", "aggregate volume", "aggregate mass", "speed(F)"]
            )
            writer.writerows(self.extrusion_tracker)

    def compute_feedrate(self):
        rect_width = self.line_width * self.delta
        return self.gamma * 60 / (self.layer_height * rect_width * self.density * 1e-6)

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
        if self.rotate:
            points_3d = np.zeros((np.array(self.points).shape[0], 3))
            points_3d[:, :2] = self.points
            self.points = bcad.rotate(
                points_3d, angle_z=np.deg2rad(-90), cnt=points_3d[0]
            )
            self.points = self.points[:, :2]
        self.points_t = np.array(self.points).T
        self.bbox = np.array(
            [
                [
                    np.min(self.points_t[0]) - self.line_width * 0.5,
                    np.min(self.points_t[1]) - self.line_width * 0.5,
                ],
                [
                    np.max(self.points_t[0]) + self.line_width * 0.5,
                    np.max(self.points_t[1]) + self.line_width * 0.5,
                ],
            ]
        )
        self.bbox_length = self.bbox[1][0] - self.bbox[0][0]
        self.bbox_width = self.bbox[1][1] - self.bbox[0][1]

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

        def distance(p0, p1):
            return np.linalg.norm(np.array(p0) - np.array(p1))

        if not self.fixed_feedrate:
            self.feedrate = self.compute_feedrate()
        self.print_length = 0
        for i in range(len(self.points)):
            current_pt = self.points[i]
            next_pt = self.points[
                (i + 1) % len(self.points)
            ]  # next point or the first point for the last one
            self.print_length += distance(current_pt, next_pt)

        self.material_consumption = (
            self.print_length
            * self.line_width
            * self.layer_height
            * self.layer_num
            * 1e-6
        )  # in Liters
        self.time_consumption = (
            self.print_length * self.layer_num / self.feedrate * 60
        )  # in seconds
        self.timedelta = timedelta(seconds=self.time_consumption)
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

        # Format hours, minutes, and seconds
        hours, remainder = divmod(self.timedelta.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        length = self.bbox_length
        width = self.bbox_width
        info_feedrate = self.feedrate
        self.gcode.append(comment(f"Timestamp: {datetime.now()}"))
        self.gcode.append(comment(f"Length: {length}"))
        self.gcode.append(comment(f"Width: {width}"))
        self.gcode.append(comment(f"Height: {self.layer_height * self.layer_num}"))
        self.gcode.append(comment(f"Layer height: {self.layer_height}"))
        self.gcode.append(comment(f"Layer number: {self.layer_num}"))
        self.gcode.append(comment(f"Line width: {self.line_width}"))
        self.gcode.append(
            comment(f"Different geometry per layer: {self.diff_geo_per_layer}")
        )
        self.gcode.append(comment(f"Tool number: {self.tool_number}"))
        self.gcode.append(comment(f"Feed rate: {info_feedrate}"))
        self.gcode.append(comment(f"Kappa: {self.kappa}"))
        self.gcode.append(comment(f"Gamma: {self.gamma}"))
        self.gcode.append(comment(f"Delta: {self.delta}"))
        self.gcode.append(comment(f"Standard: {self.standard}"))
        self.gcode.append(comment(f"Coordinate system: {self.coordinate_system}"))
        self.gcode.append(comment(f"Unit: {self.unit}"))
        self.gcode.append(comment(f"Nozzle diameter: {self.nozzle_diameter}"))

        self.gcode.append(
            comment(f"Material consumption(L): {self.material_consumption}")
        )
        self.gcode.append(
            comment(f"Estimated time consumption: {hours}hr:{minutes}min:{seconds}sec")
        )
        print(self.time_consumption)
        print(self.feedrate)
        self.gcode.append(
            comment(
                f"Original point: ({self.offset_from_origin[0]},{self.offset_from_origin[1]})"
            )
        )


class GcodeMultiplier(object):
    def __init__(
        self,
        num_horizant: int,
        num_vertic: int,
        plate_length: float,
        plate_width: float,
        gcode_params: dict,
        standard: str = "ConcretePrinter",
    ) -> None:
        self.num_horizant = num_horizant
        self.num_vertic = num_vertic
        self.plate_length = plate_length
        self.plate_width = plate_width
        self.gcode_params = gcode_params
        self.need_tune = False
        self.unit_length = 0
        self.unit_width = 0
        self.gcodelist = []
        self.grid = []
        self.visualize_cache = []
        self.standard = standard
        self.gcode_info = {}
        self.gcode_geo = {}
        self.recnst_geo = {}
        self.grid = []
        self.visualize_cache = []
        for i in range(num_horizant * num_vertic):
            self.gcodelist.append(GcodeFromPoints(**self.gcode_params))
        self.divide_plate()

    def divide_plate(self):
        unit_length = self.plate_length / self.num_horizant
        unit_width = self.plate_width / self.num_vertic
        sample_length = self.gcodelist[0].bbox_length
        sample_width = self.gcodelist[0].bbox_width
        if unit_length < sample_length or unit_width < sample_width:
            raise ValueError("The plate is too small to be divided. Perhaps rotate it?")
        else:
            self.unit_length = unit_length
            self.unit_width = unit_width
            for i in range(0, self.num_vertic):
                for j in range(0, self.num_horizant):
                    self.grid.append(
                        np.array(
                            [
                                [j * unit_length, i * unit_width],
                                [
                                    (j + 1) * unit_length,
                                    (i + 1) * unit_width,
                                ],
                            ]
                        )
                    )

    def auto_balance(self, grid: np.ndarray, points: np.ndarray):
        """Auto balance the points in the grid"""

        cnt_points = bcad.center_of_mass(points)
        cnt_grid = bcad.center_of_mass(grid)
        offset = cnt_grid - cnt_points
        return offset + points

    def keep_distance(self, dist_hori: float, dist_vert: float):
        """Keep the distance between the structures"""
        if self.num_horizant > 2:
            if self.unit_length - dist_hori < self.gcodelist[0].bbox_length:
                raise ValueError(
                    f"The horizontal distance is too large, at least {self.gcodelist[0].bbox_length - self.unit_length + dist_hori}mm short"
                )
        else:
            if self.unit_length - dist_hori * 0.5 < self.gcodelist[0].bbox_length:
                raise ValueError(
                    f"The horizontal distance is too large, at least {self.gcodelist[0].bbox_length - self.unit_length + dist_hori}mm short"
                )
        if self.num_vertic > 2:
            if self.unit_width - dist_vert < self.gcodelist[0].bbox_width:
                raise ValueError(
                    f"The vertical distance is too large, at least {self.gcodelist[0].bbox_width - self.unit_width + dist_vert}mm short"
                )
        else:
            if self.unit_width - dist_vert * 0.5 < self.gcodelist[0].bbox_width:
                raise ValueError(
                    f"The vertical distance is too large, at least {self.gcodelist[0].bbox_width - self.unit_width + dist_vert}mm short"
                )
        for i in range(0, self.num_vertic):
            new_box = np.array([[0, 0], [0, 0]])
            if i == 0:
                new_box[1][1] = -dist_vert * 0.5
            elif i == self.num_vertic - 1:
                new_box[0][1] = dist_vert * 0.5
            for j in range(0, self.num_horizant):
                if j == 0:
                    new_box[1][0] = -dist_hori * 0.5
                elif j == self.num_horizant - 1:
                    new_box[0][0] = dist_hori * 0.5
                bbox = self.gcodelist[i * self.num_horizant + j].bbox
                new_grid = self.grid[i * self.num_horizant + j] + new_box
                cnt_bbox = bcad.center_of_mass(bbox)
                cnt_grid = bcad.center_of_mass(new_grid)
                offset = cnt_grid - cnt_bbox
                self.gcodelist[i * self.num_horizant + j].points += offset
                self.visualize_cache.append(
                    bcad.bounding_box(self.gcodelist[i * self.num_horizant + j].points)
                )

    def create(
        self,
        auto_balance: bool = True,
        dist_horizont: float = 0,
        dist_vertic: float = 0,
    ):
        """Create multiple gcode files in one plate.
        Args:
            auto_balance: Auto balance the points in the grid
            dist_horizont: The distance between the structures in horizontal direction
            dist_vertic: The distance between the structures in vertical direction
        """

        if auto_balance:
            for i in range(0, self.num_vertic):
                for j in range(0, self.num_horizant):
                    self.gcodelist[
                        i * self.num_horizant + j
                    ].points = self.auto_balance(
                        self.grid[i * self.num_horizant + j],
                        self.gcodelist[i * self.num_horizant + j].points,
                    )
                    self.visualize_cache.append(
                        bcad.bounding_box(
                            self.gcodelist[i * self.num_horizant + j].points
                        )
                    )
        else:
            self.keep_distance(dist_horizont, dist_vertic)
        for ind, gcd in enumerate(self.gcodelist):
            gcode_name = Path(self.gcode_params["in_file_path"]).stem
            gcd.create(gcd.in_file_path, f"{gcode_name}_P{ind+1}.gcode")

    def read_gcode(self, in_file_path: str):
        """Read gcode from file

        :param filepath: Path to file
        :type filepath: str
        :return: gcode
        :rtype: str
        """
        in_file_path = Path(in_file_path)
        gcode_name = in_file_path.stem
        with open(in_file_path, "r", encoding="utf-8") as f:
            gcode = f.readlines()
            annotation = []
            moving_cmd = []
            geo_container = []
            info_container = {}

            for line in gcode:
                if line.startswith(";"):
                    annotation.append(line)
                elif line.startswith(self.LinearMove):
                    moving_cmd.append(line)
            for line in annotation:
                if "Different geometry per layer" in line:
                    self.different_geo_per_layer = True
                key_value = line.split(":", 1)
                if len(key_value) == 2:
                    key, value = key_value
                    key = key.strip("; \n")
                    value = value.strip()

                    # Optional: Convert value to a numeric type if applicable
                    if value.startswith("(") and value.endswith(")"):
                        # Remove parentheses and split by comma
                        value = value[1:-1].split(",")
                        # Convert each item to float
                        value = tuple(float(val.strip()) for val in value)
                    elif value.replace(".", "", 1).isdigit():
                        value = float(value) if "." in value else int(value)

                    info_container[key] = value
            self.gcode_info[gcode_name] = info_container
            ind1 = 0
            counter = 0
            for line in moving_cmd:
                condition = self.SetZ in line
                if ind1 == 1 and condition:
                    ind1 = 0
                if ind1 == 1:
                    pattern = r"([XYEF])(\d+\.\d+)"
                    matches = re.findall(pattern, line)
                    values = {letter: float(number) for letter, number in matches}
                    geo_container[-1].append(values)
                if condition:
                    ind1 = 1
                    counter += 1
                    if not self.different_geo_per_layer and counter > 1:
                        break
                    geo_container.append([])
            self.gcode_geo[gcode_name] = geo_container

    def reconstruct_geo(self):
        """Reconstruct the geometry from the gcode"""
        for geo_name, geo in self.gcode_geo.items():
            cnt = np.zeros(2)
            bbox = np.zeros((2, 2))
            layers = []
            for layer in geo:
                points = []
                extrusions = []
                for state in layer:
                    point = np.array([state[self.SetX], state[self.SetY]])
                    extrusion = (
                        state[self.LengthOfExtrude]
                        if self.LengthOfExtrude in state
                        else 0
                    )
                    points.append(point)
                    extrusions.append(extrusion)
                one_layer = {"points": points, "extrusions": extrusions}
                layers.append(one_layer)
            bbox = bcad.bounding_box(np.array([layer["points"] for layer in layers]))
            cnt = bcad.center_of_mass(bbox)
            self.recnst_geo[geo_name] = {"bbox": bbox, "cnt": cnt, "layers": layers}

    def visualize(self):
        """Visualize the gcode files in one plate"""
        # Create a figure and axis
        fig, ax = plt.subplots()

        # Create a rectangle patch
        rect = Rectangle(
            (0, 0),
            self.plate_length,
            self.plate_width,
            edgecolor="red",
            facecolor="none",
        )

        # Add the rectangle to the Axes
        ax.add_patch(rect)

        for bottom_left, upper_right in self.visualize_cache:
            # Calculate width and height
            width = upper_right[0] - bottom_left[0]
            height = upper_right[1] - bottom_left[1]

            # Create a rectangle patch
            rect = Rectangle(
                (bottom_left[0], bottom_left[1]),
                width,
                height,
                edgecolor="blue",
                facecolor="none",
            )

            # Add the rectangle to the Axes
            ax.add_patch(rect)

        # Set limits to display the rectangle
        ax.set_xlim(-1, self.plate_length + 1)
        ax.set_ylim(-1, self.plate_width + 1)

        # Display the plot
        plt.show()

    def create_multiple_gcode(
        self,
        auto_balance: bool = True,
        dist_horizont: float = 0,
        dist_vertic: float = 0,
    ):
        """Create multiple gcode files in one plate.
        Args:
            auto_balance: Auto balance the points in the grid
            dist_horizont: The distance between the structures in horizontal direction
            dist_vertic: The distance between the structures in vertical direction
        """

        if auto_balance:
            for i in range(0, self.num_vertic):
                for j in range(0, self.num_horizant):
                    self.gcodelist[
                        i * self.num_horizant + j
                    ].points = self.auto_balance(
                        self.grid[i * self.num_horizant + j],
                        self.gcodelist[i * self.num_horizant + j].points,
                    )
                    self.visualize_cache.append(
                        bcad.bounding_box(
                            self.gcodelist[i * self.num_horizant + j].points
                        )
                    )
        else:
            self.keep_distance(dist_horizont, dist_vertic)
        # for ind, gcd in enumerate(self.gcodelist):
        #     gcd.create(
        #         gcd.in_file_path, f"{self.gcode_params['standard']}_P{ind+1}.gcode"
        #     )
