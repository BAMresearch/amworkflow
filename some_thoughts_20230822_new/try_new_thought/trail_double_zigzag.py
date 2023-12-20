import numpy as np
from scipy.optimize import fsolve

from amworkflow.src.geometries.composite_geometry import CreateWallByPointsUpdate
from amworkflow.src.utils.writer import stl_writer


def zigzag_infill(
    overall_length: float = None,
    overall_width: float = None,
    line_width: float = None,
    zigzag_num: int = 1,
    angle: float = 1.1468,
    regular: bool = False,
    side_len: float = None,
):
    def cal_len_wid(variable):
        x, y = variable
        x_rad = np.radians(
            x
        )  # Convert degrees to radians for trigonometric calculations
        # eq1 = 2 * (np.cos(x_rad) + 1) * y + 1.5 * line_width - overall_length
        eq1 = (
            ((2 * np.cos(x_rad)) * y + line_width * np.sin(x_rad) * 2) * zigzag_num
            + 2 * line_width
            - overall_length
        )
        eq2 = 3 * line_width + 2 * y * np.sin(x_rad) - overall_width
        return [eq1, eq2]

    def half_zigzag(
        origin: np.ndarray, side_length1: float, side_length2, angle: float
    ) -> np.ndarray:
        """Create half of a honeycomb geometry.

        Args:
            origin: Origin of the half honeycomb.
            length: Length of the honeycomb infill.
            width: Width of the honeycomb infill.
            line_width: Width of the honeycomb lines.

        Returns:
            points: list of points defining half of the honeycomb geometry.

        """
        points = np.zeros((5, 2))
        points[0] = origin
        # points[1] = origin + np.array([0.5 * side_length, np.sqrt(3) * side_length / 2])
        # points[2] = origin + np.array([length, 0])
        # points[3] = origin + np.array([0.5 * length, -np.sqrt(3) * side_length / 2])
        # points[4] = origin + np.array([0.5 * length, 0])
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
                (x, y) for x in range(0, 89, 10) for y in range(1, overall_length, 10)
            ]
            # Set for storing unique solutions
            updated_solutions = set()
            # Loop through each initial guess to find all possible solutions
            for guess in initial_guesses:
                sol = fsolve(cal_len_wid, guess)
                # Round the solutions to 2 decimal places to avoid minor variations
                rounded_sol = (round(sol[0], 16), round(sol[1], 16))

                # Check if the solution is within the specified ranges and not already found
                if 0 <= rounded_sol[0] <= 90 and 0 <= rounded_sol[1] <= 150:
                    updated_solutions.add(rounded_sol)
            if updated_solutions:
                ideal_sol = min(updated_solutions, key=lambda x: np.abs(x[0] - 45))
            else:
                raise ValueError("No solution found")
            length = ideal_sol[1]
            angle = np.radians(ideal_sol[0])
        else:
            length = (
                (overall_length - 1.5 * line_width) / zigzag_num
            ) - 2 * line_width * np.sin(angle) / (2 * np.cos(angle))
            overall_width = 3 * line_width + 2 * np.sin(angle) * length
        start_point = np.array([0, line_width * 0.5])
        offset = np.array([line_width * 1 + line_width * np.sin(angle) * 0.5, 0])
    else:
        length = side_len
        start_point = np.array([0, line_width * 0.5])
        offset = np.array([line_width * 0.5 + length * 0.5, 0])
        regular_length = (
            1.5 * line_width + (2 * np.cos(angle) + 2) * side_len * zigzag_num
        )
        overall_length = regular_length
        overall_width = 3 * line_width + 2 * np.sin(angle) * length

    # def cal_angle(x):
    #     return np.cos(x) + 1 - np.sin(x) - 0.75 * line_width / length

    # angle = fsolve(cal_angle, 0)[0]

    # unit_width = 4 * line_width + np.sqrt(3) * length
    point_num = 12 + (zigzag_num - 1) * 10
    half_points = np.zeros((int(point_num / 2), 2))
    half_points[0] = start_point
    for i in range(zigzag_num):
        if i == 0:
            start = start_point + offset
            honeycomb_unit = half_zigzag(
                start, length, line_width * np.sin(angle), angle
            )
        else:
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
            honeycomb_unit = half_zigzag(
                start, length, line_width * np.sin(angle), angle
            )
        # if i == zigzag_num - 1:
        #     honeycomb_unit[-1][0] += line_width * 0.5
        half_points[i * 5 + 1 : i * 5 + 6] = honeycomb_unit
    another_half = half_points.copy()
    another_half[:, 1] = -another_half[:, 1]
    another_half = np.flipud(another_half)
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


th = 11

ppt = zigzag_infill(overall_length=150, overall_width=150, line_width=th, zigzag_num=1)
# 60 degree: 1.04719
# 150x150x150x10: volume(L): 1.3706792917581947
# 150x150x150x11: volume(L): 1.4895852945607906
wall = CreateWallByPointsUpdate(ppt, th, 150)
print(ppt)
wall.visualize(all_polygons=False, display_central_path=True)
wall_shape = wall.Shape()
stl_writer(
    wall_shape,
    "doublezigzag_150x150x11x150",
    store_dir="/Users/yuxianghe/Documents/BAM/amworkflow_restructure",
)
print(wall.volume)
