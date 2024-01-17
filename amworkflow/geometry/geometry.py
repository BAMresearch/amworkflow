import logging
import typing
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from OCC.Core.StlAPI import StlAPI_Writer
from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Extend.DataExchange import write_step_file, write_stl_file
from scipy.optimize import fsolve

import amworkflow.geometry.builtinCAD as bcad
from amworkflow.geometry import composite_geometries, simple_geometries

typing.override = lambda x: x


class Geometry:
    """Base class with API for any geometry creation."""

    def __init__(self, *args, **kwargs) -> None:
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)

    @typing.override
    def create(
        self, out_step: Path, out_path: Path, out_stl: Path | None = None
    ) -> None:
        """Create step and stl file from the geometry.

            To be defined in child classes.

        Args:
            out_step: File path of step file.
            out_path: File path of file containing the print path points (csv) (csv).
            out_path: File path of file containing the print path points (csv).
            out_path: File path of file containing the print path points (csv).


            out_stl: Optional file path of stl file.

        Returns:

        """
        raise NotImplementedError


class GeometryOCC(Geometry):
    def __init__(
        self,
        stl_linear_deflection: float = 0.001,
        stl_angular_deflection: float = 0.1,
        **kwargs,
    ) -> None:
        """Geometry case class for OCC geometry creation.

        Args:
            stl_linear_deflection: Linear deflection parameter of OCC stl export (Lower, more accurate mesh; OCC default is 0.001)
            stl_angular_deflection: Angular deflection parameter of OCC stl export (Lower, more accurate_mesh: OCC default is 0.5).
        """

        """Geometry case class for OCC geometry creation.

        Args:
            stl_linear_deflection: Linear deflection parameter of OCC stl export (Lower, more accurate mesh; OCC default is 0.001)
            stl_angular_deflection: Angular deflection parameter of OCC stl export (Lower, more accurate_mesh: OCC default is 0.5).
        """
        """Geometry case class for OCC geometry creation.

        Args:
            stl_linear_deflection: Linear deflection parameter of OCC stl export (Lower, more accurate mesh; OCC default is 0.001)
            stl_angular_deflection: Angular deflection parameter of OCC stl export (Lower, more accurate_mesh: OCC default is 0.5).
        """

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
        line_width: float | None = None,
        line_width: float | None = None,
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
        self.line_width = line_width
        self.line_width = line_width
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
        # assert self.radius is not None
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
            points = self.honeycomb_infill(self.length, self.width, self.line_width)
            creator = composite_geometries.CreateWallByPoints(
                points, th=self.line_width, height=self.height
            )
            shape = creator.Shape()
            points = self.honeycomb_infill(self.length, self.width, self.line_width)
            creator = composite_geometries.CreateWallByPoints(
                points, th=self.line_width, height=self.height
            )
            shape = creator.Shape()
        elif self.infill == "zigzag":
            points = self.zigzag_infill(self.length, self.width, self.line_width)
            creator = composite_geometries.CreateWallByPoints(
                points, th=self.line_width, height=self.height
            )
            shape = creator.Shape()
            points = self.zigzag_infill(self.length, self.width, self.line_width)
            creator = composite_geometries.CreateWallByPoints(
                points, th=self.line_width, height=self.height
            )
            shape = creator.Shape()
        else:
            raise ValueError(f"Unknown infill type {self.infill}")

        return shape

    def honeycomb_infill(
        self,
        overall_length: float,
        overall_width: float,
        line_width: float,
        honeycomb_num: int = 1,
        angle: float = 1.1468,
        regular: bool = False,
        side_len: float = None,
        expand_factor: float = 0,
    ):
        """
        Create honeycomb geometry.

        Args:
            overall_length: Length of the honeycomb infill.
            overall_width: Width of the honeycomb infill.
            line_width: Width of the honeycomb lines.
            honeycomb_num: Number of honeycomb units.
            angle: Angle of the honeycomb lines.
            regular: Regular honeycomb or not. A regular honeycomb has a fixed side length.
            side_len: Side length of the honeycomb.
            expand_factor: Factor to expand the honeycomb geometry.


        Returns:
            np.ndarray: Array of points defining the honeycomb geometry.
        """

        def calculate_lengths_and_widths(angle_length_pair):
            x_rad = np.radians(angle_length_pair[0])
            equation1 = (
                (2 * np.cos(x_rad) + 2) * angle_length_pair[1] * honeycomb_num
                + 1.5 * line_width
                - overall_length
            )
            equation2 = (
                3 * line_width
                + 2 * angle_length_pair[1] * np.sin(x_rad)
                - overall_width
            )
            return [equation1, equation2]

        def create_half_honeycomb(origin, side_length1, side_length2, angle):
            points = np.zeros((5, 2))
            points[0] = origin
            points[1] = origin + np.array(
                [side_length1 * np.cos(angle), side_length1 * np.sin(angle)]
            )
            points[2] = points[1] + np.array([side_length2, 0])
            points[3] = points[2] + np.array(
                [side_length1 * np.cos(angle), -side_length1 * np.sin(angle)]
            )
            points[4] = points[3] + np.array([side_length2 * 0.5, 0])
            return points

        if not regular:
            if overall_width is not None:
                overall_length -= int(line_width)
                overall_width -= int(line_width)
                initial_guesses = [
                    (x, y)
                    for x in range(0, 89, 10)
                    for y in range(1, overall_length, 10)
                ]
                updated_solutions = {
                    (round(sol[0], 10), round(sol[1], 10))
                    for guess in initial_guesses
                    for sol in [fsolve(calculate_lengths_and_widths, guess)]
                    if 0 <= sol[0] <= 90 and 0 <= sol[1] <= 150
                }
                if updated_solutions:
                    ideal_solution = min(
                        updated_solutions, key=lambda x: np.abs(x[0] - 45)
                    )
                else:
                    raise ValueError("No solution found")
                length = ideal_solution[1]
                angle = np.radians(ideal_solution[0])
            else:
                length = (
                    (overall_length - 1.5 * line_width)
                    / (2 * np.cos(angle) + 2)
                    / honeycomb_num
                )
                overall_width = 3 * line_width + 2 * np.sin(angle) * length
        else:
            length = side_len
            overall_length = (
                1.5 * line_width + (2 * np.cos(angle) + 2) * side_len * honeycomb_num
            )
            overall_width = 3 * line_width + 2 * np.sin(angle) * length

        start_point = np.array([0, line_width * 0.5])
        offset = np.array([line_width * 0.5 + length * 0.5, 0])

        half_points = np.zeros((int((12 + (honeycomb_num - 1) * 10) / 2), 2))
        half_points[0] = start_point
        for i in range(honeycomb_num):
            start = (
                start_point
                + offset
                + np.array([(2 * np.cos(angle) + 2) * length * i, 0])
            )
            honeycomb_unit = create_half_honeycomb(start, length, length, angle)
            half_points[i * 5 + 1 : i * 5 + 6] = honeycomb_unit

        another_half = np.flipud(np.copy(half_points) * [1, -1])
        points = np.concatenate((half_points, another_half), axis=0)
        outer_points = np.array(
            [
                [0, -overall_width * 0.5],
                [overall_length, -overall_width * 0.5],
                [overall_length, overall_width * 0.5],
                [0, overall_width * 0.5],
            ]
        )
        points = np.concatenate((points, outer_points), axis=0)
        if expand_factor != 0:
            cnt = np.mean(points, axis=0)
            points[2:-6] = (points[2:-6] - cnt) * (1 + expand_factor) + cnt
        return points

    def zigzag_infill(
        self,
        overall_length: float,
        overall_width: float,
        line_width: float,
        zigzag_num: int = 1,
        angle: float = 1.1468,
        regular: bool = False,
        side_len: float = None,
        expand_factor: float = 0,
    ):
        """
        Create zigzag geometry.

        Args:
            overall_length: Length of the zigzag infill.
            overall_width: Width of the zigzag infill.
            line_width: Width of the zigzag lines.
            zigzag_num: Number of zigzag units.
            angle: Angle of the zigzag lines.
            regular: Regular zigzag or not. A regular zigzag is equivalent to a diamond.
            side_len: Side length of the zigzag.
            expand_factor: Factor to expand the zigzag geometry.

        Returns:
            np.ndarray: Array of points defining the zigzag geometry.
        """

        def calculate_lengths_and_widths(angle_length_pair):
            x_rad = np.radians(angle_length_pair[0])
            eq1 = (
                (
                    (2 * np.cos(x_rad)) * angle_length_pair[1]
                    + line_width * np.sin(x_rad) * 2
                )
                * zigzag_num
                + 2 * line_width
                - overall_length
            )
            eq2 = (
                3 * line_width
                + 2 * angle_length_pair[1] * np.sin(x_rad)
                - overall_width
            )
            return [eq1, eq2]

        def create_half_zigzag(origin, side_length1, side_length2, angle):
            points = np.zeros((5, 2))
            points[0] = origin
            points[1] = origin + np.array(
                [side_length1 * np.cos(angle), side_length1 * np.sin(angle)]
            )
            points[2] = points[1] + np.array([side_length2, 0])
            points[3] = points[2] + np.array(
                [side_length1 * np.cos(angle), -side_length1 * np.sin(angle)]
            )
            points[4] = points[3] + np.array([side_length2 * 0.5, 0])
            return points

        if not regular:
            if overall_width is not None:
                overall_length -= line_width
                overall_width -= line_width
                initial_guesses = [
                    (x, y)
                    for x in range(0, 89, 10)
                    for y in range(1, int(overall_length), 10)
                ]
                updated_solutions = {
                    (round(sol[0], 16), round(sol[1], 16))
                    for guess in initial_guesses
                    for sol in [fsolve(calculate_lengths_and_widths, guess)]
                    if 0 <= sol[0] <= 90 and 0 <= sol[1] <= 150
                }
                if updated_solutions:
                    ideal_solution = min(
                        updated_solutions, key=lambda x: np.abs(x[0] - 45)
                    )
                else:
                    raise ValueError("No solution found")
                length = ideal_solution[1]
                angle = np.radians(ideal_solution[0])
            else:
                length = (
                    overall_length - 1.5 * line_width
                ) / zigzag_num - 2 * line_width * np.sin(angle) / (2 * np.cos(angle))
                overall_width = 3 * line_width + 2 * np.sin(angle) * length
        else:
            length = side_len
            overall_length = (
                1.5 * line_width + (2 * np.cos(angle) + 2) * side_len * zigzag_num
            )
            overall_width = 3 * line_width + 2 * np.sin(angle) * length

        start_point = np.array([0, line_width * 0.5])
        offset = np.array([line_width * 1 + line_width * np.sin(angle) * 0.5, 0])

        point_num = 12 + (zigzag_num - 1) * 10
        half_points = np.zeros((int(point_num / 2), 2))
        half_points[0] = start_point
        for i in range(zigzag_num):
            start = (
                start_point
                + offset
                + np.array(
                    [
                        (2 * np.cos(angle) * length + 2 * line_width * np.sin(angle))
                        * i,
                        0,
                    ]
                )
            )
            zigzag_unit = create_half_zigzag(
                start, length, line_width * np.sin(angle), angle
            )
            half_points[i * 5 + 1 : i * 5 + 6] = zigzag_unit

        another_half = np.flipud(np.copy(half_points) * [1, -1])
        points = np.concatenate((half_points, another_half), axis=0)
        outer_points = np.array(
            [
                [0, -overall_width * 0.5],
                [overall_length, -overall_width * 0.5],
                [overall_length, overall_width * 0.5],
                [0, overall_width * 0.5],
            ]
        )
        points = np.concatenate((points, outer_points), axis=0)
        if expand_factor != 0:
            cnt = np.mean(points, axis=0)
            points[2:-6] = (points[2:-6] - cnt) * (1 + expand_factor) + cnt

        return points

    def honeycomb_infill(
        self,
        overall_length: float,
        overall_width: float,
        line_width: float,
        honeycomb_num: int = 1,
        angle: float = 1.1468,
        regular: bool = False,
        side_len: float = None,
        expand_factor: float = 0,
    ):
        """
        Create honeycomb geometry.

        Args:
            overall_length: Length of the honeycomb infill.
            overall_width: Width of the honeycomb infill.
            line_width: Width of the honeycomb lines.
            honeycomb_num: Number of honeycomb units.
            angle: Angle of the honeycomb lines.
            regular: Regular honeycomb or not. A regular honeycomb has a fixed side length.
            side_len: Side length of the honeycomb.
            expand_factor: Factor to expand the honeycomb geometry.


        Returns:
            np.ndarray: Array of points defining the honeycomb geometry.
        """

        def calculate_lengths_and_widths(angle_length_pair):
            x_rad = np.radians(angle_length_pair[0])
            equation1 = (
                (2 * np.cos(x_rad) + 2) * angle_length_pair[1] * honeycomb_num
                + 1.5 * line_width
                - overall_length
            )
            equation2 = (
                3 * line_width
                + 2 * angle_length_pair[1] * np.sin(x_rad)
                - overall_width
            )
            return [equation1, equation2]

        def create_half_honeycomb(origin, side_length1, side_length2, angle):
            points = np.zeros((5, 2))
            points[0] = origin
            points[1] = origin + np.array(
                [side_length1 * np.cos(angle), side_length1 * np.sin(angle)]
            )
            points[2] = points[1] + np.array([side_length2, 0])
            points[3] = points[2] + np.array(
                [side_length1 * np.cos(angle), -side_length1 * np.sin(angle)]
            )
            points[4] = points[3] + np.array([side_length2 * 0.5, 0])
            return points

        if not regular:
            if overall_width is not None:
                overall_length -= int(line_width)
                overall_width -= int(line_width)
                initial_guesses = [
                    (x, y)
                    for x in range(0, 89, 10)
                    for y in range(1, overall_length, 10)
                ]
                updated_solutions = {
                    (round(sol[0], 10), round(sol[1], 10))
                    for guess in initial_guesses
                    for sol in [fsolve(calculate_lengths_and_widths, guess)]
                    if 0 <= sol[0] <= 90 and 0 <= sol[1] <= 150
                }
                if updated_solutions:
                    ideal_solution = min(
                        updated_solutions, key=lambda x: np.abs(x[0] - 45)
                    )
                else:
                    raise ValueError("No solution found")
                length = ideal_solution[1]
                angle = np.radians(ideal_solution[0])
            else:
                length = (
                    (overall_length - 1.5 * line_width)
                    / (2 * np.cos(angle) + 2)
                    / honeycomb_num
                )
                overall_width = 3 * line_width + 2 * np.sin(angle) * length
        else:
            length = side_len
            overall_length = (
                1.5 * line_width + (2 * np.cos(angle) + 2) * side_len * honeycomb_num
            )
            overall_width = 3 * line_width + 2 * np.sin(angle) * length

        start_point = np.array([0, line_width * 0.5])
        offset = np.array([line_width * 0.5 + length * 0.5, 0])

        half_points = np.zeros((int((12 + (honeycomb_num - 1) * 10) / 2), 2))
        half_points[0] = start_point
        for i in range(honeycomb_num):
            start = (
                start_point
                + offset
                + np.array([(2 * np.cos(angle) + 2) * length * i, 0])
            )
            honeycomb_unit = create_half_honeycomb(start, length, length, angle)
            half_points[i * 5 + 1 : i * 5 + 6] = honeycomb_unit

        another_half = np.flipud(np.copy(half_points) * [1, -1])
        points = np.concatenate((half_points, another_half), axis=0)
        outer_points = np.array(
            [
                [0, -overall_width * 0.5],
                [overall_length, -overall_width * 0.5],
                [overall_length, overall_width * 0.5],
                [0, overall_width * 0.5],
            ]
        )
        points = np.concatenate((points, outer_points), axis=0)
        if expand_factor != 0:
            cnt = np.mean(points, axis=0)
            points[2:-6] = (points[2:-6] - cnt) * (1 + expand_factor) + cnt
        return points

    def zigzag_infill(
        self,
        overall_length: float,
        overall_width: float,
        line_width: float,
        zigzag_num: int = 1,
        angle: float = 1.1468,
        regular: bool = False,
        side_len: float = None,
        expand_factor: float = 0,
    ):
        """
        Create zigzag geometry.

        Args:
            overall_length: Length of the zigzag infill.
            overall_width: Width of the zigzag infill.
            line_width: Width of the zigzag lines.
            zigzag_num: Number of zigzag units.
            angle: Angle of the zigzag lines.
            regular: Regular zigzag or not. A regular zigzag is equivalent to a diamond.
            side_len: Side length of the zigzag.
            expand_factor: Factor to expand the zigzag geometry.

        Returns:
            np.ndarray: Array of points defining the zigzag geometry.
        """

        def calculate_lengths_and_widths(angle_length_pair):
            x_rad = np.radians(angle_length_pair[0])
            eq1 = (
                (
                    (2 * np.cos(x_rad)) * angle_length_pair[1]
                    + line_width * np.sin(x_rad) * 2
                )
                * zigzag_num
                + 2 * line_width
                - overall_length
            )
            eq2 = (
                3 * line_width
                + 2 * angle_length_pair[1] * np.sin(x_rad)
                - overall_width
            )
            return [eq1, eq2]

        def create_half_zigzag(origin, side_length1, side_length2, angle):
            points = np.zeros((5, 2))
            points[0] = origin
            points[1] = origin + np.array(
                [side_length1 * np.cos(angle), side_length1 * np.sin(angle)]
            )
            points[2] = points[1] + np.array([side_length2, 0])
            points[3] = points[2] + np.array(
                [side_length1 * np.cos(angle), -side_length1 * np.sin(angle)]
            )
            points[4] = points[3] + np.array([side_length2 * 0.5, 0])
            return points

        if not regular:
            if overall_width is not None:
                overall_length -= line_width
                overall_width -= line_width
                initial_guesses = [
                    (x, y)
                    for x in range(0, 89, 10)
                    for y in range(1, int(overall_length), 10)
                ]
                updated_solutions = {
                    (round(sol[0], 16), round(sol[1], 16))
                    for guess in initial_guesses
                    for sol in [fsolve(calculate_lengths_and_widths, guess)]
                    if 0 <= sol[0] <= 90 and 0 <= sol[1] <= 150
                }
                if updated_solutions:
                    ideal_solution = min(
                        updated_solutions, key=lambda x: np.abs(x[0] - 45)
                    )
                else:
                    raise ValueError("No solution found")
                length = ideal_solution[1]
                angle = np.radians(ideal_solution[0])
            else:
                length = (
                    overall_length - 1.5 * line_width
                ) / zigzag_num - 2 * line_width * np.sin(angle) / (2 * np.cos(angle))
                overall_width = 3 * line_width + 2 * np.sin(angle) * length
        else:
            length = side_len
            overall_length = (
                1.5 * line_width + (2 * np.cos(angle) + 2) * side_len * zigzag_num
            )
            overall_width = 3 * line_width + 2 * np.sin(angle) * length

        start_point = np.array([0, line_width * 0.5])
        offset = np.array([line_width * 1 + line_width * np.sin(angle) * 0.5, 0])

        point_num = 12 + (zigzag_num - 1) * 10
        half_points = np.zeros((int(point_num / 2), 2))
        half_points[0] = start_point
        for i in range(zigzag_num):
            start = (
                start_point
                + offset
                + np.array(
                    [
                        (2 * np.cos(angle) * length + 2 * line_width * np.sin(angle))
                        * i,
                        0,
                    ]
                )
            )
            zigzag_unit = create_half_zigzag(
                start, length, line_width * np.sin(angle), angle
            )
            half_points[i * 5 + 1 : i * 5 + 6] = zigzag_unit

        another_half = np.flipud(np.copy(half_points) * [1, -1])
        points = np.concatenate((half_points, another_half), axis=0)
        outer_points = np.array(
            [
                [0, -overall_width * 0.5],
                [overall_length, -overall_width * 0.5],
                [overall_length, overall_width * 0.5],
                [0, overall_width * 0.5],
            ]
        )
        points = np.concatenate((points, outer_points), axis=0)
        if expand_factor != 0:
            cnt = np.mean(points, axis=0)
            points[2:-6] = (points[2:-6] - cnt) * (1 + expand_factor) + cnt

        return points


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
        wall_maker = composite_geometries.CreateWallByPoints(
            self.points,
            self.layer_thickness,
            self.layer_height * self.number_of_layers,
            is_close=False,
        )
        design = wall_maker.Shape()
        # design = None
        return design
