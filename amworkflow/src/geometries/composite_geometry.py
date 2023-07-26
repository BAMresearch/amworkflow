from amworkflow.src.geometries.simple_geometry import create_edge, create_wire, create_face, create_prism, random_polygon_constructor, angle_of_two_arrays, laterality_indicator, angular_bisector
from amworkflow.src.geometries.operator import reverse, geom_copy, translate, rotate_face, fuser, hollow_carver, cutter3D
from OCC.Core.gp import gp_Pnt
import numpy as np
from amworkflow.src.utils.writer import stl_writer, step_writer
from amworkflow.src.geometries.builder import geometry_builder
from amworkflow.src.geometries.mesher import get_geom_pointer
import gmsh
from amworkflow.src.utils.visualizer import mesh_visualizer
from OCC.Core.TopoDS import TopoDS_Face, TopoDS_Shell
from amworkflow.src.geometries.property import get_face_center_of_mass
from amworkflow.src.geometries.builder import sewer



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

def create_wall_by_points(pts:list, th: float, isclose:bool, height: float = None, debug: bool = False, output: str = "prism") -> np.ndarray or TopoDS_Face or TopoDS_Shell:
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
    th *= 0.5
    opt_pts = []
    vecs = []
    dir_vecs = []
    ths = []
    opt_pts_0 = []
    opt_pts_1 = []
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
        opt_pts_0.append(dr * nth + p)
        opt_pts.append(dr * nth + p)
    if isclose:
        for i,p in enumerate(pts):
            dr = -dir_vecs[i]
            nth = ths[i]
            opt_pts_1.append(dr * nth + p)
    else:
        for i,p in enumerate(pts[::-1]):
            dr = -dir_vecs[::-1][i]
            nth = ths[::-1][i]
            if debug:
                opt_pts_1.append(dr * nth + p)
            else:
                opt_pts.append(dr * nth + p)
    if debug:          
        return np.array(opt_pts_0), np.array(opt_pts_1)
    else:
        gp_pts_0 = [gp_Pnt(i[0],i[1],i[2]) for i in opt_pts_0]
        gp_pts_1 = [gp_Pnt(i[0],i[1],i[2]) for i in opt_pts_1]
        gp_pts = [gp_Pnt(i[0],i[1],i[2]) for i in opt_pts]
        if not isclose:
            poly = random_polygon_constructor(gp_pts)
        else:
            poly_o = random_polygon_constructor(gp_pts_0)
            poly_i = random_polygon_constructor(gp_pts_1)
            poly = cutter3D(poly_o, poly_i)
        match output:
            case "face":
                if not isclose:
                    return poly
                else:
                    return poly
            case "prism":
                pr = create_prism(poly, [0,0,height],True)
                return pr
        

        
    
            
        
        
    






