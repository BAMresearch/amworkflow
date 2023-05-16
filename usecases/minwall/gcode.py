"""write simple gcode based on global parameters
    see e.g. https://www.maschinfo.de/CNC-G-Code for gcode numbers
"""

import pathlib

import numpy as np


def generate_gcode_simple(
    stl: pathlib.Path | str, params: dict[str, str | float]
) -> None:
    """generate a simple gcode using the zigzag line analog to wall_zigzag.geo file

     fixed zig zag line with 3 elements
    // 4 - ----------------------------------------------------3
    // |               /6\                                  /8\|
    // |                                                       |
    // |                                                       |
    // |                                  \7/                  |
    // 5|1 - --------------------------------------------------2
    //
    toolpath: 1-2-3-4-5-6-7-8

    Args:
        params: parameter dict - has to include ("length": "width": "height": , "layer_width": "layer_height": "velocity": "feed": "machine zero point":

    Returns: saved gcode

    """

    # some variables describing the tool path analog to wall_zigzag.geo file
    hh = params["length"] - 2 * params["layer_width"]
    hh1 = hh / 3
    bb = params["width"] - 2 * params["layer_width"]
    cc = np.sqrt(hh1 * hh1 + bb * bb)
    cos_alpha = hh1 / cc
    sin_alpha = bb / cc
    dy = 0.5 * params["layer_width"] / cos_alpha

    # create points for path in x/y
    points_xy = [
        [params["layer_width"], params["layer_width"] / 2],  # 1
        [
            params["length"] - 3 / 2 * params["layer_width"],
            params["layer_width"] / 2,
        ],  # 2
        [
            params["length"] - 3 / 2 * params["layer_width"],
            params["width"] - params["layer_width"],
        ],  # 3
        [params["layer_width"] / 2, params["width"] - params["layer_width"]],  # 4
        [params["layer_width"] / 2, params["layer_width"] / 2],  # 5
        [
            params["layer_width"] + hh1,
            params["width"] - params["layer_width"] - dy / 2,
        ],  # 6
        [params["layer_width"] + 2 * hh1, params["layer_width"] + dy / 2],  # 7
        [params["layer_width"] + hh, params["width"] - params["layer_width"]],  # 8
    ]

    # change points to machine zero point
    zero_point = params["machine zero point"]
    new_points = np.array(points_xy)
    new_points[:, 0] += zero_point[0]
    new_points[:, 1] += zero_point[1]
    points_xy = new_points

    # distance between two following points for extrusion parameter
    distance = []
    for i in range(len(points_xy) - 1):
        d = np.linalg.norm(points_xy[i + 1] - points_xy[i])
        distance.append(d)

    # points in z per layer
    points_z = (
        np.arange(0, int(params["height"] / params["layer_height"])) + 1
    ) * params["layer_height"]

    # write gcode # maybe use pip package pygcode for writing
    gcode_out = stl.with_suffix(".gcode")
    with open(gcode_out, "a") as gfile:
        gfile.write(";gcode wall with infill zigzag\n")
        gfile.write(f';Layer height: {params["layer_height"]}\n')
        gfile.write(
            f';Outer dimensions (length, width, height): {params["length"]} x {params["width"]} x {params["height"]}\n'
        )
        gfile.write("G90\n")  # absolute positioning
        gfile.write("G28\n")  # Nullpunkt
        for h, idx in enumerate(points_z):
            gfile.write(f";Layer{idx}\n")
            gfile.write(f"G1 Z{h:.3f} F4800\n")  # z position
            gfile.write(
                f"G1 X{points_xy[0][0]:.3f} Y{points_xy[0][1]:.3f} F4800\n"
            )  # start position
            gfile.write("G92 E0.0000\n")  # extrusion zero
            E = 0 + distance[0] * params["feed"]
            gfile.write(
                f"G1 X{points_xy[1][0]:.3f} Y{points_xy[1][1]:.3f} E{E:.3f} F{params['velocity']*60}\n"
            )  # first movement velocity in mm/min!!
            for xy in range(2, len(points_xy)):
                E += distance[xy - 1] * params["feed"]
                gfile.write(
                    f"G1 X{points_xy[xy][0]:.3f} Y{points_xy[xy][1]:.3f} E{E:.3f}\n"
                )  # movement

        gfile.write("G28\n")  # Nullpunkt
