from amworkflow.src.geometries.simple_geometry import create_edge, create_wire, create_face, create_prism, random_polygon_constructor, angle_of_two_arrays, laterality_indicator, angular_bisector, p_center_of_mass, linear_interpolate, Pnt, Segments
from amworkflow.src.geometries.operator import reverse, geom_copy, translate, rotate_face, fuser, hollow_carver, cutter3D, bender
from amworkflow.src.geometries.property import topo_explorer, p_bounding_box
from amworkflow.src.geometries.builder import solid_maker
from OCC.Core.gp import gp_Pnt
import numpy as np
from OCC.Core.TopoDS import TopoDS_Face, TopoDS_Shell, TopoDS_Shape
from OCC.Core.BRepFilletAPI import BRepFilletAPI_MakeFillet
from amworkflow.src.geometries.property import get_face_center_of_mass
from amworkflow.src.geometries.builder import sewer
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import itertools
import copy as cp
import networkx as nx


def polygon_maker(side_num: int,
                  side_len: float,
                  rotate: float = None,
                  bound: bool = False):
    """
    @brief Creates a regular polygon. The polygon is oriented counterclockwise around the origin. If bound is True the polygon will be only the boundary (TopoDS_Wire) the polygon.
    @param side_num Number of sides of the polygon.
    @param side_len Length of the side of the polygon.
    @param rotate Rotation angle ( in radians ). Defaults to None which means no rotation.
    @param bound output only the boundary. Defaults to False. See documentation for create_wire for more information.
    @return face or boundary of the polygon.
    """
    vertices = []
    sides = []
    r = side_len / (2 * np.sin(np.pi/side_num))
    ang = np.linspace(0,2*np.pi, side_num + 1)
    if rotate != None:
        ang += rotate
    for i in range(side_num + 1):
        if i != side_num:
            x = np.sin(ang[i]) * r
            y = np.cos(ang[i]) * r
            t_pnt = gp_Pnt(x, y, 0)
            vertices.append(t_pnt)
        if i != 0:
            if i == side_num :
                t_edge = create_edge(pnt1=vertices[0], pnt2=vertices[-1])
                sides.append(t_edge)
                wire = create_wire(wire, t_edge)
            elif i == 1:
                t_edge = create_edge(pnt1=vertices[i-1], pnt2=t_pnt)
                sides.append(t_edge)
                wire = create_wire(t_edge)
            else:
                t_edge = create_edge(pnt1=vertices[i-1], pnt2=t_pnt)
                sides.append(t_edge)
                wire = create_wire(wire, t_edge)
    
    face = create_face(wire)
    if bound:
        return wire
    else:
        return reverse(face)

def hexagon_multiplier(side_num: int, side_len: float, iter_num: int, wall: float, center: gp_Pnt = None) -> TopoDS_Face:
    """
    @brief Creates a hexagon with multiplier. This is an iterative approach to the topological sorting algorithm.
    @param side_num Number of sides in the hexagon.
    @param side_len Length of the side ( s ) to be used for the multiplication.
    @param iter_num Number of iterations to perform. Default is 1.
    @param wall Wall thickness.
    @param center Center of the multiplication. Default is original point.
    @return TopoDS_Face. Note that it is the caller's responsibility to check if there is enough space
    """

    def multiplier_unit(face_odd, face_even, unit_len):
        """
         @brief Fuse a unit of mass to an odd or even side number.
         @param face_odd face with odd side number ( numpy array )
         @param face_even face with even side number ( numpy array )
         @param unit_len length of side ( numpy array )
         @return one unit of hexagon multiplication.
        """
        if side_num % 2 == 0:
            face = face_even
            ang = np.linspace(0,2*np.pi, side_num + 1)
        else:
            face = face_odd
            ang = np.linspace(np.pi / side_num,2*np.pi + np.pi / side_num, side_num + 1)
        cnt = get_face_center_of_mass(face)
        len = unit_len * 0.5 / np.tan(np.pi / side_num) * 2
        face_collect = [face]
        fuse = face
        for ag in ang:
            rot_face = rotate_face(face, ag)
            x = len * np.sin(ag)
            y = len * np.cos(ag)
            translate(rot_face, [cnt[0] + x, cnt[1] + y, 0])
            face_collect.append(rot_face)
            fuse = fuser(fuse, rot_face)
        return fuse
    # This function will generate a hollow carver for each iteration.
    for i in range(iter_num):
        # This function computes the fuse of the carver.
        if i == 0:
            face_even = hollow_carver(polygon_maker(side_num, side_len, rotate=-np.pi / side_num), wall)
            face_odd = hollow_carver(polygon_maker(side_num, side_len),wall)
            fuse = multiplier_unit(face_odd, face_even, unit_len=side_len)
        else:
            face_even = fuse
            # face_odd = rotate_face(fuse,angle = -np.pi / side_num) 
            face_odd = fuse
            fuse = multiplier_unit(face_odd, face_even, unit_len=3 * i * side_len)
    # translate center to use list of coordinates
    if isinstance(center, gp_Pnt):
        translate(fuse, list(center.Coord()))
    return fuse

def isoceles_triangle_maker(bbox_len:float, bbox_wid: float, thickness: float = None):
    """
     @brief (Having problem with wall thickness now.) Create isoceles triangulation. This is a function to create isoceles triangulation of a bounding box and its widest corner
     @param bbox_len length of bounding box of the triangle
     @param bbox_wid width of bounding box of the triangle
     @param thickness thickness of the wall of the triangle
     @return a hollowed triangle face.
    """
    h = bbox_wid
    l = bbox_len
    t = thickness if thickness != None else 0
    ary = np.array([h,l])
    r1 = np.linalg.norm(ary*0.5, 2)
    ang1 = np.arccos((h*0.5) / r1)
    ang2 = np.pi - ang1
    r = [h*0.5, r1, r1]
    thet = np.arctan(l * 0.5 / h)
    thet1 = np.arctan(h / l)
    t1 = t / np.sin(thet)
    t2 = t / np.sin(thet1)
    h1 = h - t - t1
    l1 = h1 * l / h
    ary1 = np.array([h1, l1])
    r2 = np.linalg.norm(0.5 * ary1, 2)
    ri = [0.5 * h1, r2, r2]
    ang = [0, ang2, -ang2]
    def worker(r):
        """
         @brief Creates a triangle in a polarized coordinate system.
         @param r The radius of each vertex.
         @return a triangle
        """
        pt_lst = []
        # Add a point on the plane.
        for i in range(3):
            x = r[i] * np.sin(ang[i])
            y = r[i] * np.cos(ang[i])
            pnt = gp_Pnt(x,y,0)
            pt_lst.append(pnt)
        # Create a wire for each point in the list of points.
        for i in range(3):
            # Create a wire for the i th point of the point.
            if i == 0:
                edge = create_edge(pt_lst[i], pt_lst[i+1])
                wire = create_wire(edge)
            elif i != 2:
                edge = create_edge(pt_lst[i], pt_lst[i+1])
                wire = create_wire(wire, edge)
            else:
                edge = create_edge(pt_lst[i], pt_lst[0])
                wire = create_wire(wire, edge)
        face = create_face(wire)
        return reverse(face)
    outer_face = worker(r)
    # The face to be cuttered.
    if t != 0:
        inner_face = worker(ri)
        new_face = cutter3D(outer_face, inner_face)
        return new_face
    else:
        return outer_face
    
def create_sym_hexagon1_infill(total_len: float, total_wid:float, height:float, th: float) :
    """
     @brief Create an infill pattern using symmetrical hexagon with defined len, height and numbers.
     @param total_len total length of the bounding box.
     @param total_wid total wid of the bounding box.
     @param height height of the prism. This is the same as height of the hexagon.
     @param th thickness of the wall of the hexagon.
     @return 
    """
    p0 = [0,th * 0.5]
    p1 = []

# def create_wall_by_points(pts:list, th: float, isclose:bool, height: float = None, debug: bool = False, debug_type: str = "linear", output: str = "prism", interpolate:float = None, R: float = None) -> np.ndarray or TopoDS_Face or TopoDS_Shell:
    """
     @brief Create a prism wall by points. It takes a list of points as a skeleton of a central path and then build a strip or a loop.
     @param pts list of 2D points that define the wall. The algorithm can compute points in 3D theoretically but the result may make no sense.
     @param th thickness of the wall.
     @param isclose True if the wall is closed (loop)
     @param height height of the wall if a prism is needed.
     @param debug if True output two groups of points for plotting.
     @param output selecting result intended to output. can be varied among "face" and "prism".
     @return two arrays or a face or a prism.
    """
    is_loop = False
    pts = [np.array(list(i.Coord())) if isinstance(i, gp_Pnt) else np.array(i) for i in pts]
    if R is not None:
        pts = polygon_interpolater(pts, interpolate)
        bender(pts, R)
        pts = [i for i in pts]
    th *= 0.5
    opt_pts = []
    vecs = []
    dir_vecs = []
    ths = []
    lft_pts = []
    rgt_pts = []
    segaments = []
    for i,p in enumerate(pts):
        if i != len(pts) - 1:
            a1 = pts[i+1] - pts[i]
            if i == 0:
                if isclose:
                    dr = angular_bisector(pts[-1] - p, a1)
                    # ang = angle_of_two_arrays(dir_vecs[i-1],dr)
                    ang2 = angle_of_two_arrays(laterality_indicator(p - pts[-1], True), dr)
                    ang_th = ang2
                    if ang2 > np.pi / 2:
                        dr *= -1
                        ang_th = np.pi - ang2
                    nth = np.abs(th / np.cos(ang_th))
                else:
                    dr = laterality_indicator(a1, True)
                    nth = th
            else:
                dr = angular_bisector(-vecs[i-1], a1)
                ang2 = angle_of_two_arrays(laterality_indicator(vecs[i-1], True), dr)
                ang_th = ang2
                if ang2 > np.pi / 2:
                    dr *= -1
                    ang_th = np.pi - ang2
                nth = np.abs(th / np.cos(ang_th))
        else:
            if isclose:
                a1 = pts[0] - pts[i]
                dr = angular_bisector(-vecs[i-1], a1)
                ang2 = angle_of_two_arrays(laterality_indicator(vecs[i-1], True), dr)
                ang_th = ang2
                if ang2 > np.pi / 2:
                    dr *= -1
                    ang_th = np.pi - ang2 
                nth = np.abs(th / np.cos(ang_th))
            else:
                dr = laterality_indicator(a1, True)
                nth = th
        vecs.append(a1)
        ths.append(nth)
        dir_vecs.append(dr)
        lft_pts.append(dr * nth + p)
        opt_pts.append(dr * nth + p)
    if isclose:
        lft_pts.append(lft_pts[0])
    for i,p in enumerate(pts[::-1]):
        dr = -dir_vecs[::-1][i]
        nth = ths[::-1][i]
        rgt_pts.append(dr * nth + p)
    if isclose:
        rgt_pts.append(rgt_pts[0])
        pts.append(pts[0])
    #find bounding box if bending needed
    if R is not None:
        mxpt, mnpt = p_bounding_box(lft_pts+rgt_pts)
    # Deal with overlapping sides.
    lft_pts = break_overlap(lft_pts)
    rgt_pts = break_overlap(rgt_pts)
    loops_lft_pt_i, peak_lft_pt_i = find_loop(lft_pts)
    loops_rgt_pt_i, peak_rgt_pt_i = find_loop(rgt_pts)
    if (len(loops_lft_pt_i+peak_lft_pt_i)>0) or (len(loops_rgt_pt_i+peak_rgt_pt_i)>0):
        is_loop = True
        loops_lft = index2array(loops_lft_pt_i, lft_pts)
        loops_rgt = index2array(loops_rgt_pt_i, rgt_pts)
        loops = loops_lft+loops_rgt
        # if interpolate is not None:
        #     for i in range(len(loops)):
        #         loops[i] = polygon_interpolater(loops[i], interpolate)
        #         if R is not None:
        #             bender(loops[i], mxpt, mnpt)
        loops_h = find_topology(loops)

    if debug:
        if debug_type == "polygon":
            fig, ax = plt.subplots()
            for polygon in loops:
                poly2d = np.array([[i[0], i[1]] for i in polygon])
                polygon_patch = Polygon(poly2d, closed=True, fill=False, edgecolor='black')
                ax.add_patch(polygon_patch)
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
            ax.set_xlim(-200, 200)
            ax.set_ylim(-200, 200)
            ax.set_title('Visualization of Polygons')
            plt.show()
        elif debug_type == "linear":
            plt.figure(figsize=(8, 6))  # Optional: Set the figure size
            output1 = np.array(lft_pts)
            output2 = np.array(rgt_pts)
            talist = np.array(pts).T
            toutput1 = output1.T
            toutput2 = output2.T
            x1 = talist[0]
            y1 = talist[1]
            x2 = toutput1[0]
            y2 = toutput1[1]
            x3 = toutput2[0]
            y3 = toutput2[1]
            plt.plot(x1, y1, 'bo-', label='central path')
            # Plot Group 2 points and connect with lines in red
            plt.plot(x2, y2, 'ro-', label='outer line')
            # Plot Group 3 points and connect with lines in green
            plt.plot(x3, y3, 'go-', label='inner line')
            # Add labels and a legend
            for i in range(x2.shape[0]):
                plt.text(x2[i], y2[i], str(i))
            for i in range(x3.shape[0]):
                plt.text(x3[i], y3[i], str(i))
            
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            plt.legend()
            plt.axis('equal')
            plt.grid(True)  # Optional: Add grid lines
            plt.show()
    else:
        if not is_loop:
            gp_pts_lft = [gp_Pnt(i[0],i[1],i[2]) for i in lft_pts]
            gp_pts_rgt = [gp_Pnt(i[0],i[1],i[2]) for i in rgt_pts]
            gp_pts = [gp_Pnt(i[0],i[1],i[2]) for i in opt_pts]
            poly = random_polygon_constructor(gp_pts)
        else:
            for i,v in enumerate(loops_h):
                for j,vv in enumerate(loops_h[i]):
                    loops_h[i][j] = list(loops_h[i][j])
                    for ind,k in enumerate(loops_h[i][j]):
                        loops_h[i][j][ind] = gp_Pnt(k[0], k[1], k[2])
            poly0 = random_polygon_constructor(loops_h[0][0])
            poly_r = poly0
            for i, h in enumerate(loops_h):
                if i == 0:
                    continue
                for j in h:
                    poly_c = random_polygon_constructor(j)
                    poly_r = cutter3D(poly_r, poly_c)
            poly = poly_r
        match output:
            case "face":
                if not isclose:
                    return poly
                else:
                    return poly
            case "prism":
                pr = create_prism(poly, [0,0,height],True)
                # top = geom_copy(poly)
                # translate(top,[0,0,height])
                # opt = sewer([top,pr,poly])
                # print(opt)
                pr_face = topo_explorer(pr,"face")
                pr_remake = sewer(pr_face)
                opt = solid_maker(pr_remake)
                return opt

def find_intersect(lines: np.ndarray) -> np.ndarray:
    parallel = False
    coplanarity = False
    l1, l2 = lines
    pt1, pt2 = l1
    pt3, pt4 = l2
    L1 = (pt2 - pt1) / np.linalg.norm(pt2 - pt1)
    L2 = (pt4 - pt3) / np.linalg.norm(pt4 - pt3)
    V1 = pt4 - pt1
    D1 = np.linalg.norm(V1)
    if np.isclose(np.dot(L1, L2),0) or np.isclose(np.dot(L1, L2),-1):
        parallel = True
        print("Two lines are parallel.")
        return np.full((3,1), np.nan)
    indicate = np.linalg.det(np.array([V1, L1, L2]))
    if np.abs(indicate) < 1e-8:
        coplanarity = True
    else:
        print("lines are not in the same plane.")
        return np.full((1,3), np.nan)
    if coplanarity and not parallel:
        if np.isclose(D1,0):
            return pt1
        else:
            pt5_pt4 = np.linalg.norm(np.cross(V1, L1))
            theta = np.arccos(np.dot(L1, L2))
            o_pt5 = pt5_pt4 / np.tan(theta)
            o_pt4 = pt5_pt4 / np.sin(theta)
            V1_n = V1 / D1
            cos_beta = np.dot(V1_n, L1)
            pt1_pt5 = D1 * cos_beta
            pt1_o = pt1_pt5 - o_pt5
            o = L1 * pt1_o + pt1
            return o

def index2array(ind: list, array: np.ndarray):
    real_array = []
    for i,v in enumerate(ind):
        item = []
        for j, vv in enumerate(v):
            item.append(array[vv])
        real_array.append(item)
    return real_array


def polygon_interpolater(plg: np.ndarray, step_len: float = None, num: int = None, isclose: bool = True):
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
    n = 1
    if not isclose:
        n = 2
    for i, pt in enumerate(plg):
        if i == len(plg) - n:
            break
        line = np.array([pt, plg[i+1]])
        if num is not None:
            p_num = num
        else:
            p_num = deter_dum(line)
        insert_p = linear_interpolate(line, p_num)
        new_plg = np.concatenate((new_plg[:pos+1],insert_p, new_plg[pos+1:]))
        pos +=p_num+1
    return new_plg

# def create_wall_by_points2(pts:list, th: float, isclose:bool, height: float = None, debug: bool = False, output: str = "prism", interpolate:float = None, R: float = None) -> np.ndarray or TopoDS_Face or TopoDS_Shell:
    """
     @brief Create a prism wall by points. It takes a list of points as a skeleton of a central path and then build a strip or a loop.
     @param pts list of 2D points that define the wall. The algorithm can compute points in 3D theoretically but the result may make no sense.
     @param th thickness of the wall.
     @param isclose True if the wall is closed (loop)
     @param height height of the wall if a prism is needed.
     @param debug if True output two groups of points for plotting.
     @param output selecting result intended to output. can be varied among "face" and "prism".
     @return two arrays or a face or a prism.
    """
    pts = [np.array(list(i.Coord())) if isinstance(i, gp_Pnt) else np.array(i) for i in pts]
    if R is not None:
        pts = polygon_interpolater(pts, interpolate)
        bender(pts, R)
        pts = [i for i in pts]
    th += 0.1
    th *= 0.5
    vecs = []
    dir_vecs = []
    ths = []
    lft_pts = []
    rgt_pts = []
    for i,p in enumerate(pts):
        if i != len(pts) - 1:
            a1 = pts[i+1] - pts[i]
            if i == 0:
                if isclose:
                    dr = angular_bisector(pts[-1] - p, a1)
                    # ang = angle_of_two_arrays(dir_vecs[i-1],dr)
                    ang2 = angle_of_two_arrays(laterality_indicator(p - pts[-1], True), dr)
                    ang_th = ang2
                    if ang2 > np.pi / 2:
                        dr *= -1
                        ang_th = np.pi - ang2
                    nth = np.abs(th / np.cos(ang_th))
                else:
                    dr = laterality_indicator(a1, True)
                    nth = th
            else:
                dr = angular_bisector(-vecs[i-1], a1)
                ang2 = angle_of_two_arrays(laterality_indicator(vecs[i-1], True), dr)
                ang_th = ang2
                if ang2 > np.pi / 2:
                    dr *= -1
                    ang_th = np.pi - ang2
                nth = np.abs(th / np.cos(ang_th))
        else:
            if isclose:
                a1 = pts[0] - pts[i]
                dr = angular_bisector(-vecs[i-1], a1)
                ang2 = angle_of_two_arrays(laterality_indicator(vecs[i-1], True), dr)
                ang_th = ang2
                if ang2 > np.pi / 2:
                    dr *= -1
                    ang_th = np.pi - ang2 
                nth = np.abs(th / np.cos(ang_th))
            else:
                dr = laterality_indicator(a1, True)
                nth = th
        vecs.append(a1)
        ths.append(nth)
        dir_vecs.append(dr)
        lft_pts.append(dr * nth + p)
        rgt_pts.append(-dr * nth + p)
    if isclose:
        lft_pts.append(lft_pts[0])
    if isclose:
        rgt_pts.append(rgt_pts[0])
        pts.append(pts[0])
    fc_set = []
    for i in range(len(lft_pts)-1):
        opts = lft_pts
        ipts = rgt_pts
        opt1 = gp_Pnt(opts[i][0], opts[i][1], opts[i][2])
        opt2 = gp_Pnt(opts[i+1][0], opts[i+1][1], opts[i+1][2])
        ipt2 = gp_Pnt(ipts[i][0], ipts[i][1], ipts[i][2])
        ipt1 = gp_Pnt(ipts[i+1][0], ipts[i+1][1], ipts[i+1][2])
        fc = random_polygon_constructor([opt1, opt2, ipt1, ipt2])
        fc_set.append(fc)
    face = fc_set[0]
    for fce in fc_set[1:]:
        face = fuser(face, fce)
    return create_prism(face, [0,0,height])
        
class CreateWallByPoints():
    def __init__(self, pts: list, th: float, height: float):
        self.coords = [np.array(list(i.Coord())) if isinstance(i, gp_Pnt) else np.array(i) for i in pts]
        self.height = height
        self.is_loop = False
        self.R = None
        self.overlap = False
        self.interpolate = 6
        self.th = th
        self.is_close = False
        self.vecs = []
        self.dir_vecs = []
        self.ths = []
        self.lft_coords = []
        self.rgt_coords = []
        self.loops_h = []
        self.loops = []
        self.fc_set = []
        self.poly: TopoDS_Shape
        
    def create_polygon(self):
        if not self.is_loop:
            gp_pts_lft = [gp_Pnt(i[0],i[1],i[2]) for i in self.lft_coords]
            gp_pts_rgt = [gp_Pnt(i[0],i[1],i[2]) for i in self.rgt_coords]
            self.gp_pts = gp_pts_lft + gp_pts_rgt
            self.poly = random_polygon_constructor(self.gp_pts)
        else:
            for i,v in enumerate(self.loops_h):
                for j,vv in enumerate(self.loops_h[i]):
                    self.loops_h[i][j] = list(self.loops_h[i][j])
                    for ind,k in enumerate(self.loops_h[i][j]):
                        self.loops_h[i][j][ind] = gp_Pnt(k[0], k[1], k[2])
            poly0 = random_polygon_constructor(self.loops_h[0][0])
            poly_r = poly0
            for i, h in enumerate(self.loops_h):
                if i == 0:
                    continue
                for j in h:
                    poly_c = random_polygon_constructor(j)
                    poly_r = cutter3D(poly_r, poly_c)
            self.poly = poly_r
        
    def Shape(self):
        if self.overlap:
            self.create_sides()
            for i in range(len(self.lft_coords)-1):
                opts = self.lft_coords
                ipts = self.rgt_coords
                opt1 = gp_Pnt(opts[i][0], opts[i][1], opts[i][2])
                opt2 = gp_Pnt(opts[i+1][0], opts[i+1][1], opts[i+1][2])
                ipt2 = gp_Pnt(ipts[i][0], ipts[i][1], ipts[i][2])
                ipt1 = gp_Pnt(ipts[i+1][0], ipts[i+1][1], ipts[i+1][2])
                fc = random_polygon_constructor([opt1, opt2, ipt1, ipt2])
                self.fc_set.append(fc)
            face = self.fc_set[0]
            for fce in self.fc_set[1:]:
                face = fuser(face, fce)
            return create_prism(face, [0,0,self.height])
        else:
            self.create_sides()
            self.create_loop()
            self.create_polygon()
            if np.isclose(self.height, 0):
                opt = self.poly
            else:
                pr = create_prism(self.poly, [0,0,self.height],True)
                pr_face = topo_explorer(pr,"face")
                pr_remake = sewer(pr_face)
                opt = solid_maker(pr_remake)
            return opt
        
    def create_sides(self):
        if self.R is not None:
            self.coords = polygon_interpolater(self.coords, self.interpolate)
            self.coords = bender(self.coords, self.R)
            self.coords = [i for i in self.coords]
        self.th *= 0.5
        for i,p in enumerate(self.coords):
            if i != len(self.coords) - 1:
                a1 = self.coords[i+1] - self.coords[i]
                if i == 0:
                    if self.is_close:
                        dr = angular_bisector(self.coords[-1] - p, a1)
                        # ang = angle_of_two_arrays(dir_vecs[i-1],dr)
                        ang2 = angle_of_two_arrays(laterality_indicator(p - self.coords[-1], True), dr)
                        ang_th = ang2
                        if ang2 > np.pi / 2:
                            dr *= -1
                            ang_th = np.pi - ang2
                        nth = np.abs(self.th / np.cos(ang_th))
                    else:
                        dr = laterality_indicator(a1, True)
                        nth = self.th
                else:
                    dr = angular_bisector(-self.vecs[i-1], a1)
                    ang2 = angle_of_two_arrays(laterality_indicator(self.vecs[i-1], True), dr)
                    ang_th = ang2
                    if ang2 > np.pi / 2:
                        dr *= -1
                        ang_th = np.pi - ang2
                    nth = np.abs(self.th / np.cos(ang_th))
            else:
                if self.is_close:
                    a1 = self.coords[0] - self.coords[i]
                    dr = angular_bisector(-self.vecs[i-1], a1)
                    ang2 = angle_of_two_arrays(laterality_indicator(self.vecs[i-1], True), dr)
                    ang_th = ang2
                    if ang2 > np.pi / 2:
                        dr *= -1
                        ang_th = np.pi - ang2 
                    nth = np.abs(self.th / np.cos(ang_th))
                else:
                    dr = laterality_indicator(a1, True)
                    nth = self.th
            self.vecs.append(a1)
            self.ths.append(nth)
            self.dir_vecs.append(dr)
            self.lft_coords.append(dr * nth + p)
            if self.overlap:
                self.rgt_coords.append(-dr * nth + p)
        if self.is_close:
            self.lft_coords.append(self.lft_coords[0])
        if not self.overlap:
            for i,p in enumerate(self.coords[::-1]):
                dr = -self.dir_vecs[::-1][i]
                nth = self.ths[::-1][i]
                self.rgt_coords.append(dr * nth + p)
        if self.is_close:
            self.rgt_coords.append(self.rgt_coords[0])
            self.coords.append(self.coords[0])
            
    def create_loop(self):
        if self.R is not None:
            mxpt, mnpt = p_bounding_box(self.lft_coords+self.rgt_coords)
        self.lft_coords = self.break_overlap(self.lft_coords)
        self.rgt_coords = self.break_overlap(self.rgt_coords)
        loops_lft_pt_i, peak_lft_pt_i = self.find_loop(self.lft_coords)
        loops_rgt_pt_i, peak_rgt_pt_i = self.find_loop(self.rgt_coords)
        if (len(loops_lft_pt_i+peak_lft_pt_i)>0) or (len(loops_rgt_pt_i+peak_rgt_pt_i)>0):
            self.is_loop = True
            self.loops_lft = index2array(loops_lft_pt_i, self.lft_coords)
            self.loops_rgt = index2array(loops_rgt_pt_i, self.rgt_coords)
            self.loops = self.loops_lft+self.loops_rgt
            self.loops_h = self.find_topology(self.loops)
        

    def is_include(self, point: np.ndarray, polygon: np.ndarray) -> bool:
        polygon = np.concatenate((polygon, np.array([polygon[0]])))
        ang = 0
        for i,v in enumerate(polygon):
            if i == polygon.shape[0]-1:
                break
            v1 = np.array(v-point)
            v2 = np.array(polygon[i+1]-point)
            # if np.isclose(np.linalg.norm(v1),0,1e-8) or np.isclose(np.linalg.norm(v2),0,1e-8):
            #     continue
            crt = angle_of_two_arrays(v1, v2)
            ang += crt
        if np.isclose(ang, 2*np.pi, 1e-2):
            return True
        else: return False
    
    def find_topology(self, loops: list) -> list:
        ck_lst = []
        loop_with_hierarchy = []
        topo = np.zeros(len(loops))
        for i,lp in enumerate(loops):
            p = lp[0]
            for j, llpp in enumerate(loops):
                if (i == j) or ((i,j) in ck_lst):
                    continue
                else:
                    if self.is_include(p, llpp):
                        topo[i] += 1
                    elif self.is_include(llpp[0], lp):
                        topo[j] += 1
                    ck_lst.append((i,j))
                    ck_lst.append((j,i))
        lyr_mx = int(np.max(topo))
        for i in range(lyr_mx+1):
            loop_h = [loops[j] for j,v in enumerate(topo) if v == i]
            loop_with_hierarchy.append(loop_h)
        return loop_with_hierarchy
    
    def break_overlap(self, pts: np.ndarray) -> np.ndarray:
        n_pts = np.copy(pts)
        inst_p = []
        # otr_vecs = np.array([pts[i+1] - pts[i] if i != len(pts)-1 else pts[0] - pts[i] for i in range(len(pts))])
        ck_lst = []
        m = len(pts)
        i = 0
        pos = 0
        while i < m:
            for ind,v in enumerate(pts):
                if i == m-1:
                    next_p = pts[0]
                else:
                    next_p = pts[i+1]
                if (np.linalg.norm(pts[i] - v, 1) < 1e-8) or (np.linalg.norm(next_p - v, 1) < 1e-8):
                    continue
                else:
                    comb = (i, ind, i+1)
                    if comb in ck_lst:
                        continue
                    else:
                        if i == m - 1:
                            half1 = v - pts[i]
                            half2 = pts[0] - v
                        else:
                            half1 = v - pts[i]
                            half2 = pts[i+1] - v
                        coline = np.isclose(np.linalg.norm(np.cross(half1, half2)), 0, 1e-8)
                        same_dir = np.dot(half1, half2) > 0
                        if coline and same_dir:
                            inst_p.append(v)
                        comb_f = list(itertools.permutations(cp.copy(comb)))
                        ck_lst += comb_f
            inst_p = np.array(inst_p)
            icr_pos = inst_p.shape[0]
            if icr_pos != 0:
                dist = [[np.linalg.norm(p-pts[i]), p] for p in inst_p]
                def sort_ind(lst):
                    return lst[0]
                std_inst_p = np.array([i[1] for i in sorted(dist, key=sort_ind)])
                n_pts = np.concatenate((n_pts[:pos+1], std_inst_p, n_pts[pos+1:]))
                pos += icr_pos
            i +=1
            pos+=1
            inst_p = []
        return n_pts
    
    def find_loop(self, pts: np.ndarray) -> tuple:
        pts = np.array([np.array(list(i.Coord())) if isinstance(i, gp_Pnt) else np.array(i) for i in pts])
        peak_p = []
        dim = pts.shape[0]
        distance = np.ones((dim, dim)) * np.nan
        for i, p in enumerate(pts):
            for j in range(i+1,dim,1):
                dist = np.linalg.norm(p - pts[j])
                distance[i,j] = dist
        knots = np.array(np.where(distance < 1e-6)).T
        loop_l = []
        for i in range(len(knots)-1):
            half1 = np.arange(knots[i, 0], knots[i+1, 0]+1)
            half2 = np.arange(knots[i+1, 1]+1, knots[i, 1])
            loop = np.concatenate((half1, half2))
            if loop.shape[0] < 3:
                if loop.shape[0] == 2:
                    if np.isclose(np.linalg.norm(pts[knots[i, 0]] - pts[loop[1]+2]), 0, 1e-8):
                        print(loop)
                        peak_p.append(loop[1])
                continue
            else:
                loop_l.append(loop)
        lst_loop = np.arange(knots[-1,0], knots[-1,1])
        if lst_loop.shape[0] >=3:
            loop_l.append(lst_loop)
        return loop_l, peak_p
    
    def visualize(self, plot_type: str) -> None:
        if len(self.create_sides) == 0:
            self.Shape()
        if plot_type == "polygon":
            fig, ax = plt.subplots()
            for polygon in self.loops:
                poly2d = np.array([[i[0], i[1]] for i in polygon])
                polygon_patch = Polygon(poly2d, closed=True, fill=False, edgecolor='black')
                ax.add_patch(polygon_patch)
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
            ax.set_xlim(-200, 200)
            ax.set_ylim(-200, 200)
            ax.set_title('Visualization of Polygons')
            plt.show()
        elif plot_type == "linear":
            plt.figure(figsize=(8, 6))  # Optional: Set the figure size
            output1 = np.array(self.lft_coords)
            output2 = np.array(self.rgt_coords)
            talist = np.array(self.coords).T
            toutput1 = output1.T
            toutput2 = output2.T
            x1 = talist[0]
            y1 = talist[1]
            x2 = toutput1[0]
            y2 = toutput1[1]
            x3 = toutput2[0]
            y3 = toutput2[1]
            plt.plot(x1, y1, 'bo-', label='central path')
            # Plot Group 2 points and connect with lines in red
            plt.plot(x2, y2, 'ro-', label='outer line')
            # Plot Group 3 points and connect with lines in green
            plt.plot(x3, y3, 'go-', label='inner line')
            # Add labels and a legend
            for i in range(x2.shape[0]):
                plt.text(x2[i], y2[i], str(i))
            for i in range(x3.shape[0]):
                plt.text(x3[i], y3[i], str(i))
            
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            plt.legend()
            plt.axis('equal')
            plt.grid(True)  # Optional: Add grid lines
            plt.show()
            
class CreateWallByPointsUpdate():
    def __init__(self, coords: list, th: float, height: float):
        self.coords = Pnt(coords).coords
        self.height = height
        self.R = None
        self.interpolate = 6
        self.th = th
        self.is_close = True
        self.vecs = []
        self.dir_vecs = []
        self.ths = []
        self.lft_coords = []
        self.rgt_coords = []
        self.side_coords: list
        self.create_sides()
        self.pnts = Segments(self.side_coords)
        self.G = nx.from_dict_of_lists(self.pnts.pts_digraph, create_using=nx.DiGraph)
        # self.all_loops = list(nx.simple_cycles(self.G))
        self.self_loops = nx.selfloop_edges(self.G)
        # self.all_loops = list(nx.simple_cycles(self.H)) # Dangerous! Ran out of memory.
        self.loop_generator = nx.simple_cycles(self.G)
        
    def create_sides(self):
        if self.R is not None:
            self.coords = polygon_interpolater(self.coords, self.interpolate)
            self.coords = bender(self.coords, self.R)
            self.coords = [i for i in self.coords]
        self.th *= 0.5
        for i,p in enumerate(self.coords):
            if i != len(self.coords) - 1:
                a1 = self.coords[i+1] - self.coords[i]
                if i == 0:
                    if self.is_close:
                        dr = angular_bisector(self.coords[-1] - p, a1)
                        # ang = angle_of_two_arrays(dir_vecs[i-1],dr)
                        ang2 = angle_of_two_arrays(laterality_indicator(p - self.coords[-1], True), dr)
                        ang_th = ang2
                        if ang2 > np.pi / 2:
                            dr *= -1
                            ang_th = np.pi - ang2
                        nth = np.abs(self.th / np.cos(ang_th))
                    else:
                        dr = laterality_indicator(a1, True)
                        nth = self.th
                else:
                    dr = angular_bisector(-self.vecs[i-1], a1)
                    ang2 = angle_of_two_arrays(laterality_indicator(self.vecs[i-1], True), dr)
                    ang_th = ang2
                    if ang2 > np.pi / 2:
                        dr *= -1
                        ang_th = np.pi - ang2
                    nth = np.abs(self.th / np.cos(ang_th))
            else:
                if self.is_close:
                    a1 = self.coords[0] - self.coords[i]
                    dr = angular_bisector(-self.vecs[i-1], a1)
                    ang2 = angle_of_two_arrays(laterality_indicator(self.vecs[i-1], True), dr)
                    ang_th = ang2
                    if ang2 > np.pi / 2:
                        dr *= -1
                        ang_th = np.pi - ang2 
                    nth = np.abs(self.th / np.cos(ang_th))
                else:
                    dr = laterality_indicator(a1, True)
                    nth = self.th
            self.vecs.append(a1)
            self.ths.append(nth)
            self.dir_vecs.append(dr)
            self.lft_coords.append(dr * nth + p)
            self.rgt_coords.append(-dr * nth + p)
        if self.is_close:
            self.lft_coords.append(self.lft_coords[0])
            self.rgt_coords.append(self.rgt_coords[0])
            self.rgt_coords = self.rgt_coords[::-1]
            self.coords.append(self.coords[0])
        else:
            self.rgt_coords = self.rgt_coords[::-1]
        self.side_coords = self.lft_coords + self.rgt_coords + self.lft_coords[0]
        
    def visualize(self, display_polygon: bool = True):
        # Extract the x and y coordinates and IDs
        a = self.pnts.pts_index
        x = [coord[0] for coord in a.values()]
        y = [coord[1] for coord in a.values()]
        ids = list(a.keys())  # Get the point IDs

        # Create a scatter plot in 2D
        plt.subplot(1,2,1)
        # plt.figure()
        plt.scatter(x, y)

        # Annotate points with IDs
        for i, (xi, yi) in enumerate(zip(x, y)):
            plt.annotate(f'{ids[i]}', (xi, yi), fontsize=12, ha='right')
        
        if display_polygon:
            for lp in self.loop_generator:
                if len(lp) > 2:
                    coords = [self.pnts.pts_index[i] for i in lp]
                    x = [point[0] for point in coords]
                    y = [point[1] for point in coords]
                    plt.plot(x + [x[0]], y + [y[0]], linestyle='-', marker='o')
            
        # Set labels and title
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('Points With Polygons detected')

        plt.subplot(1,2,2)
        # layout = nx.spring_layout(self.G)
        layout = nx.circular_layout(self.G)
        # Draw the nodes and edges
        nx.draw(self.G, pos=layout, with_labels=True, node_color='skyblue', font_size=10, node_size=300)
        plt.title("Multi-Digraph")
        plt.tight_layout() 
        # Show the plot
        plt.show()
    
    def get_loops(self):
        return [i for i in self.all_loops if len(i) > 2]

    def visualize_graph(self):
        layout = nx.spring_layout(self.G)
        # Draw the nodes and edges
        nx.draw(self.G, pos=layout, with_labels=True, node_color='skyblue', font_size=10, node_size=500)
        plt.title("NetworkX Graph Visualization")
        plt.show()
    