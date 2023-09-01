from amworkflow.api import amWorkflow as aw 
import numpy as np
from OCC.Core.BRepFilletAPI import BRepFilletAPI_MakeFillet
from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_MakePipe
g = aw.geom
pt1 = g.pnt(0,0,0)
pt2 = g.pnt(0,8,0)
pt3 = g.pnt(0,8,3)
pt4 = g.pnt(0,0,3)
wire1 = g.create_wire_by_points([pt1, pt2, pt3, pt4])
intersect = g.create_face(wire1)
cnt1 = g.get_face_center_of_mass(intersect)

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
p6 = g.pnt(6 * l, -(1.5*th + (np.sqrt(3) * l) * 0.5))
p7 = g.pnt(6 * l, (1.5*th + (np.sqrt(3) * l) * 0.5))
p8 = g.pnt(0, (1.5*th + (np.sqrt(3) * l) * 0.5))
pout = [p5, p6, p7, p8]
pout_nd = [i.Coord() for i in pout]
pmfo = np.vstack((pmf, pout_nd))

path = [g.pnt(i[0], i[1], i[2]) for i in pmfo]

direct = np.array(cnt1) - np.array(p0.Coord())
g.translate(intersect, direct)
path_wire = g.create_wire_by_points(path)
path_fillet = BRepFilletAPI_MakeFillet(path_wire)
edges = g.topo_explorer(path_wire, "edge")
for edge in edges:
    path_fillet.Add(0.5, edge)
path_fillet.Shape()
prism = BRepOffsetAPI_MakePipe(path_fillet, path_wire).Shape()
aw.tool.write_stl(prism, "pipe.stl", store_dir="/home/yuxiang/Documents/BAM/amworkflow/test_main/try_vertical_way")