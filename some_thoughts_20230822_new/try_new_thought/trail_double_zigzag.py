import csv
import os
from pathlib import Path

import numpy as np
from scipy.optimize import fsolve

from amworkflow.src.geometries.composite_geometry import CreateWallByPointsUpdate
from amworkflow.src.utils.writer import stl_writer


def zigzag_infill(
    overall_length: float,
    overall_width: float,
    line_width: float,
    zigzag_num: int = 1,
    angle: float = 1.1468,
    regular: bool = False,
    side_len: float = None,
):
    """
    Create zigzag geometry.

    Args:
        overall_length: Length of the zigzag infill.
        overall_width: Width of the zigzag infill.
        line_width: Width of the zigzag lines.
        ...

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
        eq2 = 3 * line_width + 2 * angle_length_pair[1] * np.sin(x_rad) - overall_width
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
                ideal_solution = min(updated_solutions, key=lambda x: np.abs(x[0] - 45))
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
                [(2 * np.cos(angle) * length + 2 * line_width * np.sin(angle)) * i, 0]
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

    return points


th = 12  # 5.659

ppt = zigzag_infill(overall_length=700, overall_width=150, line_width=th, zigzag_num=3)
# 60 degree: 1.04719
# 150x150x150x10: volume(L): 1.3706792917581947
# 150x150x150x11: volume(L): 1.4895852945607906
wall = CreateWallByPointsUpdate(ppt, th, 150)
print(ppt)
wall.visualize(all_polygons=False, display_central_path=True)
wall_shape = wall.Shape()
stl_writer(
    wall_shape,
    "double_zigzag_700x150x11.3x150",
    store_dir="/Users/yuxianghe/Documents/BAM/amworkflow_restructure",
)
print(wall.volume)
file_path = "/Users/yuxianghe/Documents/BAM/amworkflow_restructure/beam_zigzag_700x150x150x12.csv"
with open(file_path, "w", newline="") as file:
    writer = csv.writer(file)

    # Writing the header
    writer.writerow(["x", "y"])

    # Writing the data
    writer.writerows(ppt)

