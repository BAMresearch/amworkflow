from amworkflow.src.geometries.composite_geometry import CreateWallByPointsUpdate
import numpy as np
from amworkflow.api import amWorkflow as aw
import cv2 as cv

th = 8
l = 10
height = 10
g = aw.geom
hth = th * 0.5
l = 20
display = True
p0 = g.pnt(0, hth, 0)
p1 = g.pnt(l * 0.5, hth)
p2 = g.pnt(l, (np.sqrt(3) * l) * 0.5 + hth)
p3 = g.pnt(2 * l, (np.sqrt(3) * l) * 0.5 + hth)
p4 = g.pnt(5 * l * 0.5, hth)
pu = [p0, p1, p2, p3, p4]
alist = np.array([list(i.Coord()) for i in pu])
put1 = g.p_translate(pu, [3 * l, 0, 0])
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
pmr = g.p_rotate(pm, angle_z=np.pi)
# pmr = g.p_translate(pmr, np.array([-th,0,0]))
cnt2 = g.p_center_of_mass(pmr)
t_len = cnt2[1] * 2
pmrt = g.p_translate(pmr, [0, -t_len, 0])
pm_lt = np.vstack((alist, put1))
pm_lt = np.vstack((pm_lt, np.array(end_p)))
pmf = np.vstack((pm_lt, pmrt))
p5 = g.pnt(0, -(1.5*th + (np.sqrt(3) * l) * 0.5))
p6 = g.pnt(6 * l + th, -(1.5*th + (np.sqrt(3) * l) * 0.5))
p7 = g.pnt(6 * l + th, (1.5*th + (np.sqrt(3) * l) * 0.5))
p8 = g.pnt(0, (1.5*th + (np.sqrt(3) * l) * 0.5))
pout = [p5, p6, p7, p8]
pout_nd = [i.Coord() for i in pout]
pmfo = np.vstack((pmf, pout_nd))

wall = CreateWallByPointsUpdate(pmfo, th, height)
lft_coords = wall.lft_coords
rgt_coords = wall.rgt_coords
pieces = []
for i in range(len(lft_coords)-1):
    pieces.append([lft_coords[i], lft_coords[i+1], rgt_coords[i+1], rgt_coords[i]])

def create_canvas(width: float, height: float):
    image = np.zeros((height, width), dtype=np.uint8)
    return image

def create_poly(pnts: list):
    vertices = np.array(pnts, np.int32)
    vertices = vertices.reshape((-1, 1, 2))
    return vertices

def add_poly(image: np.ndarray, poly: np.ndarray):
    cv.fillPoly(image, [poly], 255)

def find_contours(image):
    contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    return contours
    
# poly = wall.Shape()
# wall.visualize(all_polygons=False, display_central_path=False)
# aw.tool.write_stl(poly, "sucess_new_scheme",store_dir="/home/yhe/Documents/new_am2/amworkflow/some_thoughts_20230822_new/try_new_thought")
image = create_canvas(150, 150)
for p in pieces:
    poly = create_poly(p)
    add_poly(image, poly)
contours = np.array(find_contours(image))[0].reshape(-1, 2)
print(contours)
# contour_image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
# cv.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
# cv.imshow("Contours", contour_image)
# cv.waitKey(0)
# cv.destroyAllWindows()