import csv

import numpy as np
from scipy.optimize import fsolve

from amworkflow.api import amWorkflow as aw
from amworkflow.src.geometries.composite_geometry import CreateWallByPointsUpdate
from amworkflow.src.utils.writer import stl_writer

def honeycomb_infill(
    overall_length: float,
    overall_width: float,
    line_width: float,
    honeycomb_num: int = 1,
    angle: float = 1.1468,
    regular: bool = False,
    side_len: float = None,
):
    """
    Create honeycomb geometry.

    Args:
        overall_length: Length of the honeycomb infill.
        overall_width: Width of the honeycomb infill.
        line_width: Width of the honeycomb lines.

    Returns:
        np.ndarray: Array of points defining the honeycomb geometry.
    """

    def calculate_lengths_and_widths(angle_length_pair):
        x_rad = np.radians(angle_length_pair[0])
        equation1 = ((2 * np.cos(x_rad) + 2) * angle_length_pair[1] * honeycomb_num + 1.5 * line_width - overall_length)
        equation2 = 3 * line_width + 2 * angle_length_pair[1] * np.sin(x_rad) - overall_width
        return [equation1, equation2]

    def create_half_honeycomb(origin, side_length1, side_length2, angle):
        points = np.zeros((5, 2))
        points[0] = origin
        points[1] = origin + np.array([side_length1 * np.cos(angle), side_length1 * np.sin(angle)])
        points[2] = points[1] + np.array([side_length2, 0])
        points[3] = points[2] + np.array([side_length1 * np.cos(angle), -side_length1 * np.sin(angle)])
        points[4] = points[3] + np.array([side_length2 * 0.5, 0])
        return points

    if not regular:
        if overall_width is not None:
            overall_length -= int(line_width)
            overall_width -= int(line_width)
            initial_guesses = [(x, y) for x in range(0, 89, 10) for y in range(1, overall_length, 10)]
            updated_solutions = {(round(sol[0], 10), round(sol[1], 10)) for guess in initial_guesses for sol in [fsolve(calculate_lengths_and_widths, guess)] if 0 <= sol[0] <= 90 and 0 <= sol[1] <= 150}
            if updated_solutions:
                ideal_solution = min(updated_solutions, key=lambda x: np.abs(x[0] - 45))
            else:
                raise ValueError("No solution found")
            length = ideal_solution[1]
            angle = np.radians(ideal_solution[0])
        else:
            length = ((overall_length - 1.5 * line_width) / (2 * np.cos(angle) + 2) / honeycomb_num)
            overall_width = 3 * line_width + 2 * np.sin(angle) * length
    else:
        length = side_len
        overall_length = 1.5 * line_width + (2 * np.cos(angle) + 2) * side_len * honeycomb_num
        overall_width = 3 * line_width + 2 * np.sin(angle) * length

    start_point = np.array([0, line_width * 0.5])
    offset = np.array([line_width * 0.5 + length * 0.5, 0])

    half_points = np.zeros((int((12 + (honeycomb_num - 1) * 10) / 2), 2))
    half_points[0] = start_point
    for i in range(honeycomb_num):
        start = start_point + offset + np.array([(2 * np.cos(angle) + 2) * length * i, 0])
        honeycomb_unit = create_half_honeycomb(start, length, length, angle)
        half_points[i * 5 + 1 : i * 5 + 6] = honeycomb_unit

    another_half = np.flipud(np.copy(half_points) * [1, -1])
    points = np.concatenate((half_points, another_half), axis=0)
    outer_points = np.array([[0, -overall_width * 0.5], [overall_length, -overall_width * 0.5], [overall_length, overall_width * 0.5], [0, overall_width * 0.5]])
    points = np.concatenate((points, outer_points), axis=0)

    return points


th = 11.4  # 5.652
# ppt = honeycomb_infill(150, 8, 3)
# ppt = honeycomb_infill(
#     regular=True, side_len=63.26, angle=np.deg2rad(84.79), honeycomb_num=2, line_width=8
# )
ppt = honeycomb_infill(
    overall_length=700, overall_width=150, line_width=th, honeycomb_num=3
)
print(ppt)
# Writing to a CSV file
file_path = "/Users/yuxianghe/Documents/BAM/amworkflow_restructure/beam_honeycomb_700x150x150x11.4"
with open(file_path, "w", newline="") as file:
    writer = csv.writer(file)

    # Writing the header
    writer.writerow(["x", "y"])

    # Writing the data
    writer.writerows(ppt)
# 60 degree: 1.04719
# 150x150x150x10: volume (L): 1.520399999948474
# 700x150x10x150: volume (L): 5.014000002962625
wall = CreateWallByPointsUpdate(ppt, th, 150)
print(ppt)
wall.visualize(all_polygons=False, display_central_path=True)
wall_shape = wall.Shape()
stl_writer(
    wall_shape,
    "honeycomb_700x150x10x150",
    store_dir="/Users/yuxianghe/Documents/BAM/amworkflow_restructure",
)
print("volume (L):", wall.volume)
# lft_coords = wall.lft_coords
# rgt_coords = wall.rgt_coords
# pieces = []
# for i in range(len(lft_coords)-1):
#     pieces.append([lft_coords[i], lft_coords[i+1], rgt_coords[i+1], rgt_coords[i]])

# def create_canvas(width: float, height: float):
#     image = np.zeros((height, width), dtype=np.uint8)
#     return image

# def create_poly(pnts: list):
#     vertices = np.array(pnts, np.int32)
#     vertices = vertices.reshape((-1, 1, 2))
#     return vertices

# def add_poly(image: np.ndarray, poly: np.ndarray):
#     cv.fillPoly(image, [poly], 255)

# def find_contours(image):
#     contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

#     return contours

# poly = wall.Shape()
# wall.visualize(all_polygons=False, display_central_path=False)
# aw.tool.write_stl(poly, "sucess_new_scheme",store_dir="/home/yhe/Documents/new_am2/amworkflow/some_thoughts_20230822_new/try_new_thought")
# image = create_canvas(150, 150)
# for p in pieces:
#     poly = create_poly(p)
#     add_poly(image, poly)
# contours = np.array(find_contours(image))[0].reshape(-1, 2)
# print(contours)
# contour_image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
# cv.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
# cv.imshow("Contours", contour_image)
# cv.waitKey(0)
# cv.destroyAllWindows()
