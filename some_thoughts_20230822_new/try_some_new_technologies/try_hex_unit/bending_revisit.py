from amworkflow.src.geometries.operator import get_occ_bounding_box, translate
from amworkflow.src.geometries.builder import geometry_builder
import numpy as np
import gmsh
from amworkflow.src.geometries.mesher import get_geom_pointer
from amworkflow.src.geometries.simple_geometry import create_prism, create_box
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common
from amworkflow.src.geometries.property import get_volume_center_of_mass
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeSolid
from OCC.Core.gp import gp_Pnt
import meshio
from amworkflow.api import amWorkflow as aw
import numpy as np
import matplotlib.pyplot as plt
from amworkflow.src.geometries.composite_geometry import find_loop
from amworkflow.src.geometries.composite_geometry import create_wall_by_points2


g = aw.geom
th = 10
hth = th * 0.5
l = 20
display = True
p0 = g.pnt(0,hth, 0)
p1 = g.pnt(l * 0.5, hth)
p2 = g.pnt(l, (np.sqrt(3) * l) * 0.5 + hth)
p3 = g.pnt(2 * l, (np.sqrt(3) * l) * 0.5 + hth)
p4 = g.pnt(5 * l * 0.5, hth)
pu = [p0, p1, p2, p3, p4]
alist = np.array([list(i.Coord()) for i in pu])
put1 = g.p_translate(pu, [3 * l, 0, 0])
end_p = np.copy(put1[-1])
end_p[0] += l * 0.5
pm = pu + put1
pm.append(end_p)

pmr = g.p_rotate(pm, angle_z=np.pi)
cnt2 = g.p_center_of_mass(pmr)
t_len = cnt2[1] * 2
pmrt = g.p_translate(pmr, [0, -t_len, 0])
pm_lt = np.vstack((alist, put1))
pm_lt = np.vstack((pm_lt, np.array(end_p)))
pmf = np.vstack((pm_lt, pmrt))
p5 = g.pnt(0, -(1.5*th + (np.sqrt(3) * l) * 0.5))
p6 = g.pnt(6 * l+th, -(1.5*th + (np.sqrt(3) * l) * 0.5))
p7 = g.pnt(6 * l+th, (1.5*th + (np.sqrt(3) * l) * 0.5))
p8 = g.pnt(0, (1.5*th + (np.sqrt(3) * l) * 0.5))
pout = [p5, p6, p7, p8]
pout_nd = [i.Coord() for i in pout]
pmfo = np.vstack((pmf, pout_nd))

# pmfo = g.p_translate(pmfo, np.array([50,50,0]))

def linear_interpolate(pts: np.ndarray, num: int): 
    for i, pt in enumerate(pts):
        if i == len(pts)-1:
            break
        else:
            interpolated_points = np.linspace(pt, pts[i+1], num=num+2)[1:-1]
    return interpolated_points

def polygon_interpolater(plg: np.ndarray, step_len: float):
    def deter_dum(line: np.ndarray):
        ratio = step_len / np.linalg.norm(line[0]-line[1])
        if ratio > 0.75:
            num = 0
        elif (ratio > 0.4) and (ratio <= 0.75):
            num = 1
        elif (ratio > 0.3) and (ratio <= 0.4):
            num = 2
        elif (ratio > 0.22) and (ratio <= 0.3):
            num = 3
        elif (ratio > 0.19) and (ratio <= 0.22):
            num = 4
        elif (ratio > 0.14) and (ratio <= 0.19):
            num = 5
        elif ratio <=0.14:
            num = 7
        return num
    new_plg = plg
    pos = 0
    for i, pt in enumerate(plg):
        if i == len(plg) - 1:
            break
        line = np.array([pt, plg[i+1]])
        num = deter_dum(line)
        insert_p = linear_interpolate(line, num)
        new_plg = np.concatenate((new_plg[:pos+1],insert_p, new_plg[pos+1:]))
        pos +=num+1
    return new_plg


comb = create_wall_by_points2(pmfo, th, True, 8, False, "prism", 7, 300)

# modified_mesh = meshio.Mesh(points=point_cordinates, cells={k: v for k, v in mesh.cells.items()})
# meshio.write('comb_R300.vtk', mesh, file_format='vtk')



# bend_new_cent_path = bender(new_cent_path)

# comb = g.create_wall_by_points(bend_new_cent_path, th, True, 30, True)


aw.tool.write_stl(comb, "hex_unit_inter_direct", store_dir="/home/yhe/Documents/new_am2/amworkflow/test_main_20230822_new/test_main/try_hex_unit")


# gmsh.initialize()
# model = gmsh.model()
# gmsh.option.setNumber("General.NumThreads",8)
# ptr = get_geom_pointer(model=model, shape=comb)
# gmsh.model.occ.synchronize()
# model.mesh.remove_duplicate_nodes()
# model.mesh.remove_duplicate_elements()
# gmsh.option.setNumber("Mesh.MeshSizeFactor", 1)
# model.mesh.generate()
# gmsh.write("comb.vtk")
# mesh = meshio.read("comb.vtk")
# point_cordinates = mesh.points

def bender(point_cordinates):
    coord_t = np.array(point_cordinates).T
    mx_pt = np.max(coord_t,1)
    mn_pt = np.min(coord_t,1)
    cnt = 0.5 * (mn_pt + mx_pt)
    scale = np.abs(mn_pt-mx_pt)
    radius = 300
    o_y = scale[1]*0.5 + radius
    for pt in point_cordinates:
        xp = pt[0]
        yp = pt[1]
        ratio_l = xp / scale[0]
        ypr = scale[1] * 0.5 - yp
        Rp = radius + ypr
        ly = scale[0] * (1 + ypr / radius)
        lp = ratio_l * ly
        thetp = lp / (Rp)
        thetp = lp / (Rp)
        pt[0] = Rp * np.sin(thetp)
        pt[1] = o_y - Rp*np.cos(thetp)

def shape_bender(point_cordinates):
    coord_t = np.array(point_cordinates).T
    mx_pt = np.max(coord_t,1)
    mn_pt = np.min(coord_t,1)
    cnt = 0.5 * (mn_pt + mx_pt)
    scale = np.abs(mn_pt-mx_pt)
    print(scale)
    radius = 300
    o_y = scale[1]*0.5 + radius
    for pt in point_cordinates:
        xp = pt[0]
        yp = pt[1]
        ratio_l = xp / scale[0]
        ypr = scale[1] * 0.5 - yp
        Rp = radius + ypr
        ly = scale[0] * (1 + ypr / radius)
        # lp = ratio_l * ly
        lp = xp
        thetp = lp / (Rp)
        thetp = lp / (Rp)
        pt[0] = Rp * np.sin(thetp)
        pt[1] = o_y - Rp*np.cos(thetp)
