import logging
import typing
from pathlib import Path

import numpy as np

import amworkflow.gcode.utils as ut
import amworkflow.geometry.builtinCAD as bcad


class Gcode:
    """Base class with API for any gcode writer."""

    def __init__(self, *args, **kwargs) -> None:
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self.width = 0
        self.height = 0
        self.nozzle_diameter = 0
        self.feedrate = 0
        self.kappa = 1
        self.tool_number = 0
        self.layer_num = 0
        self.layer_height = 0
        self.coordinate_system = "Absolute"
        self.gcd_writer = ut.GcodeCommand()

    @typing.override
    def create(self, in_file: Path, out_gcode: Path) -> None:
        """Create gcode file by given path file or geometry file

        Args:
            in_file: File path to path point file or stl file from geometry step
            out_gcode File path of output gcode file.

        Returns:

        """
        pt1 = bcad.Pnt([2, 4])
        pt2 = bcad.Pnt([3, 5])
        pt3 = bcad.Pnt([4, 6])
        pt4 = bcad.Pnt([5, 7])
        wire = [pt1, pt2, pt3, pt4]
        self.gcd_writer.init_gcode()
        for i in range(self.layer_num):
            self.gcd_writer.elevate(i * self.layer_height, self.feedrate)
            for ind, pt in enumerate(wire):
                if ind == 0:
                    self.gcd_writer.move(pt, 0, self.feedrate)
                else:
                    e = self.compute_extrusion(wire[ind - 1], pt)
                    self.gcd_writer.move(pt, e)
        self.gcd_writer.write_gcode(out_gcode, self.gcd_writer.gcode)

        # raise NotImplementedError

    def compute_extrusion(self, p0: bcad.Pnt, p1: bcad.Pnt):
        """Compute the extrusion length. rectify the extrusion length by the kappa factor.

        :param p0: The previous point
        :type p0: bcad.Pnt
        :param p1: The current point
        :type p1: bcad.Pnt
        :return: The extrusion length
        :rtype: float
        """
        self.nozzle_area = 0.25 * np.pi * self.nozzle_diameter**2
        L = bcad.distance(p0, p1)
        E = np.round(L * self.width * self.height / self.nozzle_area, 4)
        if self.kappa == 0:
            logging.warning("Kappa is zero, set to 1")
            self.kappa = 1
        return E / self.kappa
