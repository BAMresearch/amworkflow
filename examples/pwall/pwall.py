from amworkflow.api import amWorkflow as aw
import numpy as np
@aw.engine.amworkflow()
def geometry_spawn(pm):
#This is where to define your model.
    th = pm.thickness
    l = pm.length
    height = pm.height*3
    g = aw.geom
    hth = th * 0.5
    display = True
    p0 = g.pnt(0, hth, 0)
    p1 = g.pnt(l * 0.5, hth)
    p2 = g.pnt(l, (np.sqrt(3) * l) * 0.5 + hth)
    p3 = g.pnt(2 * l, (np.sqrt(3) * l) * 0.5 + hth)
    p4 = g.pnt(5 * l * 0.5, hth)
    pu = [p0, p1, p2, p3, p4] #one unit of the points _/-\
    alist = np.array([list(i.Coord()) for i in pu]) # get the coord from the points
    put1 = g.p_translate(pu, [3 * l, 0, 0]) # Translate the unit _/-\_/-\
    end_p = np.copy(put1[-1]) 
    end_p[0] += l * 0.5 # Add one point to make half of the infill _/-\_/-\_
    p_up = pu + put1 # integrate the points together
    p_up.append(end_p) # add the point
    # pm_cnt = g.p_center_of_mass(pm)
    # pm_cnt[0] -=hth
    pmr = g.p_rotate(p_up, angle_z=np.pi) # Rotate the half infill to make it upside down
    # pmr = g.p_translate(pmr, np.array([-th,0,0]))
    cnt2 = g.p_center_of_mass(pmr) #Get the center of mass of all points
    t_len = cnt2[1] * 2 #2 times the y coord would be the length for translation
    pmrt = g.p_translate(pmr, [0, -t_len, 0])
    pm_lt = np.vstack((alist, put1))
    pm_lt = np.vstack((pm_lt, np.array(end_p)))
    pmf = np.vstack((pm_lt, pmrt))
    p5 = g.pnt(0, -(1.5*th + (np.sqrt(3) * l) * 0.5)) # create points for the outerline
    p6 = g.pnt(6 * l + th, -(1.5*th + (np.sqrt(3) * l) * 0.5))
    p7 = g.pnt(6 * l + th, (1.5*th + (np.sqrt(3) * l) * 0.5))
    p8 = g.pnt(0, (1.5*th + (np.sqrt(3) * l) * 0.5))
    pout = [p5, p6, p7, p8]
    pout_nd = [i.Coord() for i in pout]
    pmfo = np.vstack((pmf, pout_nd))
    pmfo_cnt = g.p_center_of_mass(pmfo)
    # pmfo = g.p_rotate(pmfo, angle_z=np.pi, cnt = pmfo_cnt)
    # pmfo = g.p_translate(pmfo, np.array([6 * l + th,0,0]))
    wall_maker = g.CreateWallByPoints(pmfo, th, height)
    wall_maker.is_close = True
    # wall_maker.visualize("linear")
    p = wall_maker.Shape()
    return p#TopoDS_Shape

# Info    :   1st: [3536, 12, 23374] #15
# Info    :   2nd: [3536, 12, 13031] #8
# Info    : The dihedral angle between them is 6.0792e-05 degree.
# Info    : Hint: You may use Mesh.AngleToleranceFacetOverlap to decrease the dihedral angle tolerance 0.1 (degree)
# Error   : Invalid boundary mesh (overlapping facets) on surface 15 surface 8