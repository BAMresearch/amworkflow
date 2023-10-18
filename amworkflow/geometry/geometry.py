import typing
from pathlib import Path
import logging
import numpy as np
from OCC.Core.StlAPI import StlAPI_Writer
from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Extend.DataExchange import write_step_file, write_stl_file

from amworkflow.geometry import composite_geometries, simple_geometries

typing.override = lambda x: x


class Geometry:
    """Base class with API for any geometry creation."""

    def __init__(self, *args, **kwargs) -> None:
        self.logger = logging.getLogger(__name__+"."+self.__class__.__name__)

    @typing.override
    def create(
        self, out_step: Path, out_path: Path, out_stl: Path | None = None
    ) -> None:
        """Create step and stl file from the geometry.

            To be defined in child classes.

        Args:
            out_step: File path of step file.
            out_path: File path of file containing the print path points.
            out_stl: Optional file path of stl file.

        Returns:

        """
        raise NotImplementedError


class GeometryOCC(Geometry):
    """Geometry base class for OCC geometry creation."""

    def __init__(
        self,
        stl_linear_deflection: float = 0.001,
        stl_angular_deflection: float = 0.1,
        **kwargs,
    ) -> None:

        self.stl_linear_deflection = stl_linear_deflection
        self.stl_angular_deflection = stl_angular_deflection

        super().__init__(**kwargs)

    @typing.override
    def geometry_spawn(self) -> TopoDS_Shape:
        """Define geometry using occ fct or own.

            To be overwritten by user.

        Args:

        Returns:
            shape: OCC shape.
        """
        raise NotImplementedError

    def create(
        self, out_step: Path, out_path: Path, out_stl: Path | None = None
    ) -> None:
        """Create step and stl file from OCC shape geometry created by function geometry_spawn.

        Args:
            out_step: File path of step file.
            out_path: File path of file containing the print path points.
            out_stl: File path of step file.

        Returns:

        """
        self.shape = self.geometry_spawn()

        # stl_write = StlAPI_Writer()
        # stl_write.SetASCIIMode(True)  # Set to False for binary STL output
        # status = stl_write.Write(self.shape, str(out_stl))
        # print(status)
        # if status:
        #     print("Done!")

        if out_stl:
            write_stl_file(
                self.shape,
                str(out_stl),
                mode="binary",
                linear_deflection=self.stl_linear_deflection,
                angular_deflection=self.stl_angular_deflection,
            )

        write_step_file(a_shape=self.shape, filename=str(out_step))

        # TODO: save print path points


class GeometryParamWall(GeometryOCC):
    def __init__(
        self,
        height: float | None = None,
        length: float | None = None,
        width: float | None = None,
        radius: float | None = None,
        infill: typing.Literal["solid", "honeycomb", "zigzag"] | None = None,
        layer_thickness: float | None = None,
        **kwargs,
    ) -> None:
        """OCC geometry class for parametric wall element with following parameters:

        Args:
            height: Height of wall.
            length: Length of wall.
            width: Width of wall.
            radius: Radius of wall curvature.
            infill: Infill of wall: possible solid, honeycomb, zigzag.
            layer_thickness: Thickness of the layers.

        """

        self.height = height
        self.length = length
        self.width = width
        self.radius = radius
        self.infill = infill
        self.layer_thickness = layer_thickness

        super().__init__(**kwargs)

    def geometry_spawn(self) -> TopoDS_Shape:
        """Define wall geometry based on given parameters.

        Args:

        Returns:
            shape: OCC shape.

        """

        assert self.height is not None
        assert self.length is not None
        assert self.width is not None
        assert self.radius is not None
        assert self.infill is not None
        # assert self.layer_thickness is not None

        if self.infill == "solid":
            shape = simple_geometries.create_box(
                length=self.length,
                width=self.width,
                height=self.height,
                radius=self.radius,
            )

        elif self.infill == "honeycomb":
            raise NotImplementedError

        elif self.infill == "zigzag":
            raise NotImplementedError

        else:
            raise ValueError(f"Unknown infill type {self.infill}")

        return shape


class GeometryCenterline(GeometryOCC):
    def __init__(
        self,
        points: np.ndarray | None = None,
        layer_thickness: float | None = None,
        number_of_layers: int | None = None,
        layer_height: float | None = None,
        **kwargs,
    ) -> None:

        """OCC geometry class for creating a layer by layer geometry from a given array of centerline points (x,y,z)
        and following paramters:

        Args:
            points: Array with the centerline points with shape (n_points, 3).
            layer_thickness: Thickness of the layers.
            number_of_layers: Number of layers.
            layer_height: Height of the layers.


        """
        self.points = points
        self.layer_thickness = layer_thickness
        self.number_of_layers = number_of_layers
        self.layer_height = layer_height

        super().__init__(**kwargs)

    def geometry_spawn(self) -> TopoDS_Shape:
        """Define geometry from centerline points
        Returns:

        """
        assert self.points is not None
        assert self.layer_thickness is not None
        assert self.number_of_layers is not None
        assert self.layer_height is not None

        # where to put those geometry building function?
        # is_close as global parameter?
        # wall_maker = CreateWallByPointsUpdate("", self.layer_thickness, self.layer_height*self.number_of_layers, is_close=False)
        # design = wall_maker.Shape()
        design = None
        return design
