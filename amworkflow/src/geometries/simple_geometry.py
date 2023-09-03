from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Wire, TopoDS_Shell, TopoDS_Solid, TopoDS_Face, TopoDS_Edge, topods_Compound
from OCC.Core.Geom import Geom_TrimmedCurve
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Ax2, gp_Dir, gp_Pln
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace, BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid, BRepBuilderAPI_MakeShell, brepbuilderapi_Precision, BRepBuilderAPI_MakePolygon
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism, BRepPrimAPI_MakeCylinder
from amworkflow.src.geometries.operator import geom_copy, translate, reverse
from OCCUtils.Construct import make_face
from amworkflow.src.geometries.builder import geometry_builder, sewer

from OCC.Core.GC import GC_MakeArcOfCircle
import math as m
from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_MakeOffsetShape
from OCCUtils.Topology import Topo
import numpy as np


def create_box(length: float,
               width: float,
               height: float,
               radius: float = None,
               alpha: float = None,
               shell: bool = False) -> TopoDS_Shape:
    """
    @brief Create a box with given length width height and radius. If radius is None or 0 the box will be sewed by a solid.
    @param length Length of the box in points
    @param width Width of the box.
    @param height Height of the box.
    @param radius Radius of the box. Default is None which means that the box is without curves.
    @param alpha defines the angle of bending the box. Default is half the length divided by the radius.
    @param shell If True the box will be shell. Default is False.
    @return TopoDS_Shape with box in it's topolar form. Note that this is a Shape
    """
    if (radius == None) or (radius == 0):
        if shell:
            box = BRepPrimAPI_MakeBox(length, width, height).Shape()
            faces = list(Topo(TopoDS_Solid).faces_from_solids(box))
            print(isinstance(faces[0], TopoDS_Shape))
            sewed_face = sewer(faces)
            # make_shell = BRepBuilderAPI_MakeShell(sewed_face, False).Shell()

            return sewed_face
        else:
            return BRepPrimAPI_MakeBox(length, width, height).Shape()
    else:
        # The alpha of the circle.
        if alpha == None:
            alpha = (length / radius) % (m.pi * 2)
        R = radius + (width / 2)
        r = radius - (width / 2)
        p1 = gp_Pnt(0, 0, 0)
        p1_2 = gp_Pnt((1 - m.cos(0.5 * alpha)) * R, R * m.sin(0.5 * alpha), 0)
        p2 = gp_Pnt((1 - m.cos(alpha)) * R, R * m.sin(alpha), 0)
        p3 = gp_Pnt(R - r * m.cos(alpha), r * m.sin(alpha), 0)
        p3_4 = gp_Pnt(R - r * m.cos(0.1 * alpha), r * m.sin(0.1 * alpha), 0)
        p4 = gp_Pnt(width, 0, 0)
        arch1_2 = GC_MakeArcOfCircle(p1, p1_2, p2)
        arch3_4 = GC_MakeArcOfCircle(p3, p3_4, p4)
        arch_edge1_2 = BRepBuilderAPI_MakeEdge(arch1_2.Value()).Edge()
        arch_edge3_4 = BRepBuilderAPI_MakeEdge(arch3_4.Value()).Edge()
        edge2 = BRepBuilderAPI_MakeEdge(p2, p3).Edge()
        edge4 = BRepBuilderAPI_MakeEdge(p4, p1).Edge()
        wire = BRepBuilderAPI_MakeWire(
            arch_edge1_2, edge2, arch_edge3_4, edge4).Wire()
        wire_top = geom_copy(wire)
        translate(wire_top, [0, 0, height])
        prism = create_prism(wire, [0, 0, height], True)
        bottom_face = create_face(wire)
        top_face = reverse(create_face(wire_top))
        component = [prism, top_face, bottom_face]
        sewing = BRepBuilderAPI_Sewing()
        # Add all components to the sewing.
        for i in range(len(component)):
            sewing.Add(component[i])
        sewing.Perform()
        sewed_shape = sewing.SewedShape()
        # shell = BRepBuilderAPI_MakeShell(sewed_shape)
        solid = BRepBuilderAPI_MakeSolid(sewed_shape).Shape()
        curve_box = geometry_builder(component)
        # Returns the shape of the shell.
        if shell:
            return sewed_shape
        else:
            return solid


def create_cylinder(radius: float, length: float) -> TopoDS_Shape:
    """
     @brief Create a cylinder shape. This is a convenience function for BRepPrimAPI_MakeCylinder
     @param radius Radius of the cylinder in metres
     @param length Length of the cylinder in metres.
     @return Shape of the cylinder ( TopoDS_Shape ) that is created and ready to be added to topology
    """
    return BRepPrimAPI_MakeCylinder(radius, length).Shape()


def create_prism(shape: TopoDS_Shape,
                 vector: list,
                 copy: bool = True) -> TopoDS_Shell:
    """
    @brief Create prism from TopoDS_Shape and vector. It is possible to copy the based wire(s) if copy is True. I don't know what if it's False so it is recommended to always use True.
    @param shape TopoDS_Shape to be used as base
    @param vector list of 3 elements ( x y z ). Normally only use z to define the height of the prism.
    @param copy boolean to indicate if the shape should be copied
    @return return the prism
    """
    return BRepPrimAPI_MakePrism(shape, gp_Vec(vector[0],
                                               vector[1],
                                               vector[2]),
                                 copy).Shape()


def create_prism_by_curve(shape: TopoDS_Shape, curve: TopoDS_Wire):
    return BRepPrimAPI_MakePrism(shape, curve).Shape()


def create_face(wire: TopoDS_Wire) -> TopoDS_Face:
    """
     @brief Create a BRep face from a TopoDS_Wire. This is a convenience function to use : func : ` BRepBuilderAPI_MakeFace ` and
     @param wire The wire to create a face from. Must be a TopoDS_Wire.
     @return A Face object with the properties specified
    """
    return BRepBuilderAPI_MakeFace(wire).Face()


def create_wire(*edge) -> TopoDS_Wire:
    """
     @brief Create a wire. Input at least one edge to build a wire. This is a convenience function to call BRepBuilderAPI_MakeWire with the given edge and return a wire.
     @return A wire built from the given edge ( s ). The wire may be used in two ways : 1
    """
    return BRepBuilderAPI_MakeWire(*edge).Wire()


def create_edge(pnt1: gp_Pnt = None, pnt2: gp_Pnt = None, arch: Geom_TrimmedCurve = None) -> TopoDS_Edge:
    """
     @brief Create an edge between two points. This is a convenience function to be used in conjunction with : func : ` BRepBuilderAPI_MakeEdge `
     @param pnt1 first point of the edge
     @param pnt2 second point of the edge
     @param arch arch edge ( can be None ). If arch is None it will be created from pnt1 and pnt2
     @return an edge.
    """
    if isinstance(pnt1, gp_Pnt) and isinstance(pnt2, gp_Pnt):
        edge = BRepBuilderAPI_MakeEdge(pnt1, pnt2).Edge()
    elif isinstance(arch, Geom_TrimmedCurve):
        edge = BRepBuilderAPI_MakeEdge(arch).Edge()
    return edge


def create_arch(pnt1, pnt2, pnt1_2, make_edge: bool = True) -> TopoDS_Edge:
    """
     @brief Create an arc of circle. If make_edge is True the arc is created in TopoDS_Edge.
     @param pnt1 The first point of the arc.
     @param pnt2 The second point of the arc.
     @param pnt1_2 The intermediate point of the arc.
     @param make_edge If True the arc is created in the x - y plane.
     @return arch : return an ` GC_MakeArcOfCircle` object or an edge
    """
    arch = GC_MakeArcOfCircle(pnt1, pnt1_2, pnt2).Value()
    # Create an edge if make_edge is true.
    if make_edge:
        return create_edge(arch)
    else:
        return arch


def create_wire_by_points(points: list):
    """
     @brief Create a closed wire (loop) by points. The wire is defined by a list of points which are connected by an edge.
     @param points A list of points. Each point is a gp_Pnt ( x y z) where x, y and z are the coordinates of a point.
     @return A wire with the given points connected by an edge. This will be an instance of : class : `BRepBuilderAPI_MakeWire`
    """
    pts = points
    # Create a wire for each point in the list of points.
    for i in range(len(pts)):
        # Create a wire for the i th point.
        if i == 0:
            edge = create_edge(pts[i], pts[i+1])
            wire = create_wire(edge)
        # Create a wire for the given points.
        if i != len(pts)-1:
            edge = create_edge(pts[i], pts[i+1])
            wire = create_wire(wire, edge)
        else:
            edge = create_edge(pts[i], pts[0])
            wire = create_wire(wire, edge)
    return wire


def random_polygon_constructor(points: list, isface: bool = True) -> TopoDS_Face or TopoDS_Wire:
    """
     @brief Creates a polygon in any shape. If isface is True the polygon is made face - oriented otherwise it is wires
     @param points List of points defining the polygon
     @param isface True if you want to create a face - oriented
     @return A polygon 
    """
    pb = BRepBuilderAPI_MakePolygon()
    # Add points to the points.
    for pt in points:
        pb.Add(pt)
    pb.Build()
    pb.Close()
    # Create a face or a wire.
    if isface:
        return create_face(pb.Wire())
    else:
        return pb.Wire()


def angle_of_two_arrays(a1: np.ndarray, a2: np.ndarray, rad: bool = True) -> float:
    """
     @brief Returns the angle between two vectors. This is useful for calculating the rotation angle between a vector and another vector
     @param a1 1D array of shape ( n_features )
     @param a2 2D array of shape ( n_features )
     @param rad If True the angle is in radians otherwise in degrees
     @return Angle between a1 and a2 in degrees or radians depending on rad = True or False
    """
    dot = np.dot(a1, a2)
    norm = np.linalg.norm(a1)*np.linalg.norm(a2)
    cos_value = np.round(dot / norm, 15)
    if rad:
        return np.arccos(cos_value)
    else:
        return np.rad2deg(np.arccos(cos_value))


def laterality_indicator(a: np.ndarray, d: bool):
    """
     @brief Compute laterality indicator of a vector. This is used to create a vector which is perpendicular to the based vector on its left side ( d = True ) or right side ( d = False )
     @param a vector ( a )
     @param d True if on left or False if on right
     @return A vector.
    """
    z = np.array([0, 0, 1])
    # cross product of z and a
    if d:
        na = np.cross(z, a)
    else:
        na = np.cross(-z, a)
    norm = np.linalg.norm(na, na.shape[0])
    return na / norm


def angular_bisector(a1: np.ndarray, a2: np.ndarray) -> np.ndarray:
    """
     @brief Angular bisector between two vectors. The result is a vector splitting the angle between two vectors uniformly.
     @param a1 1xN numpy array
     @param a2 1xN numpy array
     @return the bisector vector
    """
    norm1 = np.linalg.norm(a1)
    norm2 = np.linalg.norm(a2)
    bst = a1 / norm1 + a2 / norm2
    norm3 = np.linalg.norm(bst)
    # The laterality indicator a2 norm3 norm3
    if norm3 == 0:
        opt = laterality_indicator(a2, True)
    else:
        opt = bst / norm3
    return opt


def p_translate(pts: np.ndarray, direct: np.ndarray) -> np.ndarray:
    pts = np.array([np.array(list(i.Coord())) if isinstance(
        i, gp_Pnt) else np.array(i) for i in pts])
    pts = [i + direct for i in pts]
    return list(pts)


def p_center_of_mass(pts: np.ndarray) -> np.ndarray:
    pts = np.array([np.array(list(i.Coord())) if isinstance(
        i, gp_Pnt) else np.array(i) for i in pts])

    return np.mean(pts.T, axis=1)


def p_rotate(pts: np.ndarray, angle_x: float = 0, angle_y: float = 0, angle_z: float = 0, cnt: np.ndarray = None) -> np.ndarray:
    pts = np.array([np.array(list(i.Coord())) if isinstance(
        i, gp_Pnt) else np.array(i) for i in pts])
    com = p_center_of_mass(pts)
    if cnt is None:
        cnt = np.array([0, 0, 0])
    t_vec = cnt - com
    pts += t_vec
    rot_x = np.array([[1, 0, 0],
                      [0, np.cos(angle_x), -np.sin(angle_x)],
                      [0, np.sin(angle_x), np.cos(angle_x)]])
    rot_y = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                      [0, 1, 0],
                      [-np.sin(angle_y), np.cos(angle_y), 0]])
    rot_z = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                      [np.sin(angle_z), np.cos(angle_z), 0],
                      [0, 0, 1]])
    R = rot_x@rot_y@rot_z
    rt_pts = pts@R
    r_pts = rt_pts - t_vec
    return r_pts

def create_face_by_plane(pln: gp_Pln, *vt: gp_Pnt) -> TopoDS_Face:
    return make_face(pln, *vt)

def linear_interpolate(pts: np.ndarray, num: int): 
    for i, pt in enumerate(pts):
        if i == len(pts)-1:
            break
        else:
            interpolated_points = np.linspace(pt, pts[i+1], num=num+2)[1:-1]
    return interpolated_points

def random_pnt_gen(xmin, xmax, ymin, ymax, zmin = 0, zmax = 0):
    random_x = np.random.randint(xmin, xmax)
    random_y = np.random.randint(ymin, ymax)
    if zmin == 0 and zmax == 0:
        random_z = 0
    else:
        random_z = np.random.randint(zmin, zmax)
    return np.array([random_x, random_y, random_z])

def random_line_gen(xmin, xmax, ymin, ymax, zmin = 0, zmax = 0):
    pt1 = random_pnt_gen(xmin, xmax,ymin,ymax,zmin,zmax)
    pt2 = random_pnt_gen(xmin, xmax,ymin,ymax,zmin,zmax)
    return np.array([pt1, pt2])
