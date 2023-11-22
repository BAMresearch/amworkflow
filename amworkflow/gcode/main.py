import logging
import typing
from pathlib import Path
import numpy as np
import amworkflow.geometry.builtinCAD as bcad


class Gcode:
    """Base class with API for any gcode writer."""

    def __init__(self, *args, **kwargs) -> None:
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self.width = 0
        self.height = 0
        self.nozzle_diameter = 0
        self.nozzle_area = 0.25 * np.pi * self.nozzle_diameter ** 2
        self.kappa = 1

    @typing.override
    def create(self, in_file: Path, out_gcode: Path) -> None:
        """Create gcode file by given path file or geometry file

        Args:
            in_file: File path to path point file or stl file from geometry step
            out_gcode File path of output gcode file.

        Returns:

        """
        raise NotImplementedError

    def compute_extrusion(self, p0: bcad.Pnt, p1: bcad.Pnt):
        """Compute the extrusion length. rectify the extrusion length by the kappa factor.

        :param p0: The previous point
        :type p0: bcad.Pnt
        :param p1: The current point
        :type p1: bcad.Pnt
        :return: The extrusion length
        :rtype: float
        """        
        L = bcad.distance(p0, p1)
        E = L * self.width * self.height / self.nozzle_area
        if self.kappa == 0:
            logging.warning("Kappa is zero, set to 1")  
            self.kappa = 1
        return E / self.kappa