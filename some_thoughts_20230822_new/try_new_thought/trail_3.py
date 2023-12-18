import numpy as np

from amworkflow.api import amWorkflow as aw
from amworkflow.src.geometries.composite_geometry import CreateWallByPointsUpdate

# th = 8
# l = 10
# height = 10
# g = aw.geom
# hth = th * 0.5
# l = 20
# display = True
# p0 = g.pnt(0, hth, 0)
# p1 = g.pnt(l * 0.5, hth)
# p2 = g.pnt(l, (np.sqrt(3) * l) * 0.5 + hth)
# p3 = g.pnt(2 * l, (np.sqrt(3) * l) * 0.5 + hth)
# p4 = g.pnt(5 * l * 0.5, hth)
# pu = [p0, p1, p2, p3, p4]
# alist = np.array([list(i.Coord()) for i in pu])
# put1 = g.p_translate(pu, [3 * l, 0, 0])
# # for i in range(len(put1)):
# #     if i == 0:
# #         continue
# #     put1[i][0] -=hth
# end_p = np.copy(put1[-1])
# end_p[0] += l * 0.5
# pm = pu + put1
# pm.append(end_p)
# # pm_cnt = g.p_center_of_mass(pm)
# # pm_cnt[0] -=hth
# pmr = g.p_rotate(pm, angle_z=np.pi)
# # pmr = g.p_translate(pmr, np.array([-th,0,0]))
# cnt2 = g.p_center_of_mass(pmr)
# t_len = cnt2[1] * 2
# pmrt = g.p_translate(pmr, [0, -t_len, 0])
# pm_lt = np.vstack((alist, put1))
# pm_lt = np.vstack((pm_lt, np.array(end_p)))
# pmf = np.vstack((pm_lt, pmrt))
# p5 = g.pnt(0, -(1.5*th + (np.sqrt(3) * l) * 0.5))
# p6 = g.pnt(6 * l + th, -(1.5*th + (np.sqrt(3) * l) * 0.5))
# p7 = g.pnt(6 * l + th, (1.5*th + (np.sqrt(3) * l) * 0.5))
# p8 = g.pnt(0, (1.5*th + (np.sqrt(3) * l) * 0.5))
# pout = [p5, p6, p7, p8]
# pout_nd = [i.Coord() for i in pout]
# pmfo = np.vstack((pmf, pout_nd))



def honeycomb_infill(
    overall_length: float, line_width: float, honeycomb_num: int = 1
) -> np.ndarray:
    """Create honeycomb geometry.

    Args:
        length: Length of the honeycomb infill.
        width: Width of the honeycomb infill.
        line_width: Width of the honeycomb lines.

    Returns:
        points: list of points defining the honeycomb geometry.

    """

    def half_honeycomb(origin: np.ndarray, side_length: float) -> np.ndarray:
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
        points[1] = origin + np.array([0.5 * side_length, np.sqrt(3) * side_length / 2])
        points[2] = points[1] + np.array([side_length, 0])
        points[3] = points[2] + np.array([0.5 * side_length, -np.sqrt(3) * side_length / 2])
        points[4] = points[3] + np.array([0.5 * side_length, 0])
        return points

    length = (overall_length - (3 * line_width)) / (3 * honeycomb_num)
    start_point = np.array([0, line_width * 0.5])
    offset = np.array([1.5 * line_width, 0])
    overall_width = 3 * line_width + np.sqrt(3) * length
    point_num = 16 + (honeycomb_num - 1) * 10
    half_points = np.zeros((int(point_num / 2) - 2, 2))
    half_points[0] = start_point
    for i in range(honeycomb_num):
        if i == 0:
            start = start_point + offset
            honeycomb_unit = half_honeycomb(start, length)
        else:
            start = start_point + offset + np.array([3 * length * i, 0])
            honeycomb_unit = half_honeycomb(start, length)
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


ppt = honeycomb_infill(150, 8, 1)


wall = CreateWallByPointsUpdate(ppt, 8, 2)
wall.visualize(all_polygons=False, display_central_path=True)
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