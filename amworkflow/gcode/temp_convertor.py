import os

import numpy as np

import amworkflow.geometry.builtinCAD as bcad


def pnt(x: float, y: float, z: float = 0):
    """Create a point"""
    return np.array([x, y, z])


def command(line: str):
    """Write one line of command to gcode file"""
    return line + "\n"


def distance(p1: np.ndarray, p2: np.ndarray):
    """Calculate the distance between two points"""
    return np.linalg.norm(p1 - p2)


def move(x: float, y: float, e: float = None):
    """Move to a point in space"""
    if e is None:
        return command(f"G1 X{x} Y{y}")
    return command(f"G1 X{x} Y{y} E{e}")


def write_gcode(filename: str, gcode: str):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(gcode)


def hexagon_infill(th, l):
    hth = th * 0.5
    l = 20
    p0 = pnt(0, hth, 0)
    p1 = pnt(l * 0.5, hth)
    p2 = pnt(l, (np.sqrt(3) * l) * 0.5 + hth)
    p3 = pnt(2 * l, (np.sqrt(3) * l) * 0.5 + hth)
    p4 = pnt(5 * l * 0.5, hth)
    pu = [p0, p1, p2, p3, p4]
    alist = np.array(pu)
    put1 = bcad.translate(pu, [3 * l, 0, 0])
    # for i in range(len(put1)):
    #     if i == 0:
    #         continue
    #     put1[i][0] -=hth
    end_p = np.copy(put1[-1])
    end_p[0] += l * 0.5
    pm = pu + put1
    pm.append(end_p)
    # pm_cnt = g.p_center_of_mass(pm)
    # pm_cnt[0] -=hth
    pmr = bcad.rotate(pm, angle_z=np.pi)
    # pmr = g.p_translate(pmr, np.array([-th,0,0]))
    cnt2 = bcad.center_of_mass(pmr)
    t_len = cnt2[1] * 2
    pmrt = bcad.translate(pmr, [0, -t_len, 0])
    pm_lt = np.vstack((alist, put1))
    pm_lt = np.vstack((pm_lt, np.array(end_p)))
    pmf = np.vstack((pm_lt, pmrt))
    p5 = pnt(0, -(1.5 * th + (np.sqrt(3) * l) * 0.5))
    p6 = pnt(6 * l + th, -(1.5 * th + (np.sqrt(3) * l) * 0.5))
    p7 = pnt(6 * l + th, (1.5 * th + (np.sqrt(3) * l) * 0.5))
    p8 = pnt(0, (1.5 * th + (np.sqrt(3) * l) * 0.5))
    pout = [p5, p6, p7, p8]
    pmfo = np.vstack((pmf, pout))
    pmfo = np.array(bcad.translate(pmfo, [l * 3, l * 3, 0]))
    return pmfo


head = "G90\nM82\nM106 S0\nM104\nS0\nT0"
output_str = ""
tail = "M104 S0\nM140 S0\nM84"


def main(l, height, num):
    global output_str
    z = 0
    output_str = command(head)
    for i in range(num):
        z += height
        coordinates = hexagon_infill(8, l)[:, :2]

        z_move = f"G1 Z{z} F3000"
        extrusion_reset = "G92 E0"
        output_str += command(z_move)
        output_str += command(extrusion_reset)
        extrusion_ratio = 0.0041
        E = 0
        for j, coord in enumerate(coordinates):
            if j == 0:
                m_command = move(coord[0], coord[1])
            else:
                dist = distance(coord, coordinates[j - 1])
                extrusion_length = dist * extrusion_ratio
                m_command = move(coord[0], coord[1], extrusion_length)
                E += extrusion_length
            output_str += command(m_command)
    output_str += command(tail)
    write_path = os.path.join(os.getcwd(), "output.gcode")
    write_gcode(write_path, output_str)


if __name__ == "__main__":
    main(20, 3, 5)
