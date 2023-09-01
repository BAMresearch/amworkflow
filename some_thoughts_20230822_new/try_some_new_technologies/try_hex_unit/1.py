
from amworkflow.api import amWorkflow as aw
import numpy as np
import matplotlib.pyplot as plt
from amworkflow.src.geometries.composite_geometry import find_loop

g = aw.geom
th = 8
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
pmfo_cnt = g.p_center_of_mass(pmfo)
# pmfo = g.p_rotate(pmfo, angle_z=np.pi, cnt = pmfo_cnt)
# pmfo = g.p_translate(pmfo, np.array([6 * l + th,0,0]))
p = g.create_wall_by_points(pmfo, th, True, 30, True, "linear", "prism")
aw.tool.write_stl(p, "hex_unit_new",
                  store_dir="/home/yhe/Documents/new_am2/amworkflow/test_main_20230822_new/test_main/try_hex_unit")
# print(find_loop(pmfo))
