import logging
import math as m

import numpy as np
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BOPAlgo import BOPAlgo_Splitter
from OCC.Core.BRep import BRep_Builder
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCC.Core.BRepBndLib import brepbndlib_Add
from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_Copy,
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakePolygon,
    BRepBuilderAPI_MakeShell,
    BRepBuilderAPI_MakeSolid,
    BRepBuilderAPI_MakeWire,
    BRepBuilderAPI_Sewing,
    BRepBuilderAPI_Transform,
    brepbuilderapi_Precision,
)
from OCC.Core.BRepGProp import brepgprop_SurfaceProperties, brepgprop_VolumeProperties
from OCC.Core.BRepPrimAPI import (
    BRepPrimAPI_MakeBox,
    BRepPrimAPI_MakeCylinder,
    BRepPrimAPI_MakePrism,
)
from OCC.Core.GC import GC_MakeArcOfCircle
from OCC.Core.Geom import Geom_TrimmedCurve
from OCC.Core.gp import gp_Ax1, gp_Dir, gp_Pln, gp_Pnt, gp_Trsf, gp_Vec
from OCC.Core.GProp import GProp_GProps
from OCC.Core.TopAbs import (
    TopAbs_COMPOUND,
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_SHELL,
    TopAbs_SOLID,
    TopAbs_WIRE,
)
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopoDS import (
    TopoDS_Compound,
    TopoDS_Edge,
    TopoDS_Face,
    TopoDS_Iterator,
    TopoDS_Shape,
    TopoDS_Shell,
    TopoDS_Solid,
    TopoDS_Wire,
    topods_Face,
)
from OCC.Core.TopTools import TopTools_ListOfShape
from OCCUtils.Construct import make_face, vec_to_dir
from OCCUtils.Topology import Topo

from amworkflow.config.settings import LOG_LEVEL

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("amworkflow.occ_helpers")
logger.setLevel(LOG_LEVEL)


def create_box(
    length: float,
    width: float,
    height: float,
    radius: float = None,
    alpha: float = None,
    shell: bool = False,
) -> TopoDS_Shape:
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
            sewed_face = sew_face(faces)
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
        wire = BRepBuilderAPI_MakeWire(arch_edge1_2, edge2, arch_edge3_4, edge4).Wire()
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
        print(solid)
        curve_box = create_compound(component)

        # Returns the shape of the shell.
        if shell:
            return sewed_shape
        else:
            return solid


def create_prism(shape: TopoDS_Shape, vector: list, copy: bool = True) -> TopoDS_Shell:
    """
    @brief Create prism from TopoDS_Shape and vector. It is possible to copy the based wire(s) if copy is True. I don't know what if it's False so it is recommended to always use True.
    @param shape TopoDS_Shape to be used as base
    @param vector list of 3 elements ( x y z ). Normally only use z to define the height of the prism.
    @param copy boolean to indicate if the shape should be copied
    @return return the prism
    """
    return BRepPrimAPI_MakePrism(
        shape, gp_Vec(vector[0], vector[1], vector[2]), copy
    ).Shape()


def create_wire(*edge) -> TopoDS_Wire:
    """
    @brief Create a wire. Input at least one edge to build a wire. This is a convenience function to call BRepBuilderAPI_MakeWire with the given edge and return a wire.
    @return A wire built from the given edge ( s ). The wire may be used in two ways : 1
    """
    return BRepBuilderAPI_MakeWire(*edge).Wire()


def create_face(wire: TopoDS_Wire) -> TopoDS_Face:
    """
    @brief Create a BRep face from a TopoDS_Wire. This is a convenience function to use : func : ` BRepBuilderAPI_MakeFace ` and
    @param wire The wire to create a face from. Must be a TopoDS_Wire.
    @return A Face object with the properties specified
    """
    return BRepBuilderAPI_MakeFace(wire).Face()


def create_edge(
    pnt1: gp_Pnt = None, pnt2: gp_Pnt = None, arch: Geom_TrimmedCurve = None
) -> TopoDS_Edge:
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


def sew_face(*component) -> TopoDS_Shape:
    sewing = BRepBuilderAPI_Sewing()
    for i in range(len(component[0])):
        sewing.Add(component[0][i])
    sewing.Perform()
    sewed_shape = sewing.SewedShape()
    return sewed_shape


def create_solid(item: TopoDS_Shape) -> TopoDS_Shape:
    return BRepBuilderAPI_MakeSolid(item).Shape()


def create_compound(*args):
    builder = BRep_Builder()
    obj = TopoDS_Compound()
    builder.MakeCompound(obj)
    for item in args[0]:
        builder.Add(obj, item)
    return obj


def translate(item: TopoDS_Shape, vector: list):
    """
    @brief Translates the shape by the distance and direction of a given vector.
    @param item The item to be translated. It must be a TopoDS_Shape
    @param vector The vector to translate the object by. The vector has to be a list with three elements
    """
    ts_handler = gp_Trsf()
    ts_handler.SetTranslation(gp_Vec(vector[0], vector[1], vector[2]))
    loc = TopLoc_Location(ts_handler)
    item.Move(loc)


def reverse(item: TopoDS_Shape):
    """
    @brief Reverse the shape.
    @param item The item to reverse.
    @return The reversed item
    """
    return item.Reversed()


def geom_copy(item: TopoDS_Shape):
    """
    @brief Copy a geometry to a new shape. This is a wrapper around BRepBuilderAPI_Copy and can be used to create a copy of a geometry without having to re - create the geometry in the same way.
    @param item Geometry to be copied.
    @return New geometry that is a copy of the input geometry.
    """
    wire_top_builder = BRepBuilderAPI_Copy(item)
    wire_top_builder.Perform(item, True)
    new_item = wire_top_builder.Shape()
    return new_item


def split(item: TopoDS_Shape, *tools: TopoDS_Shape) -> TopoDS_Compound:
    top_list = TopTools_ListOfShape()
    for i in tools:
        top_list.Append(i)
    cut = BOPAlgo_Splitter()
    print(tools)
    cut.SetArguments(top_list)
    cut.Perform()
    return cut.Shape()


def split_by_plane(
    item: TopoDS_Shape,
    nz: int = None,
    layer_height: float = None,
    nx: int = None,
    ny: int = None,
):
    """
    @brief Split a TopoDS_Shape into sub - shapes.
    @param item TopoDS_Shape to be split.
    @param nz Number of layers in z - points to split.
    @param layer_height Layer height ( m ).
    @param nx Number of sub - shapes in the x - direction.
    @param ny Number of sub - shapes in the y - direction.
    @return a compound of sub-shapes
    """

    xmin, ymin, zmin, xmax, ymax, zmax = get_occ_bounding_box(item)
    plan_len = 1.2 * max(abs(xmin - xmax), abs(ymin - ymax))
    z = zmax - zmin
    if nz is not None and layer_height is not None:
        if not np.isclose(z, nz * layer_height, atol=1e-3):
            raise ValueError(
                f"Only one of nz or layer_height can be specified. If both are specified, the product of nz: {nz} and layer_height: {layer_height} must be equal to the height of the shape: {z}."
            )
        else:
            logger.warning(
                "Only one of nz or layer_height can be specified. You have specified both but the product of nz: %s and layer_height: %s is equal to the height of the shape: %s. I am just going to use nz for splitting the mesh."
                % (nz, layer_height, z)
            )
    if nz is not None:
        z_list = np.linspace(zmin, z, nz + 1)
    elif layer_height is not None:
        z_list = np.arange(zmin, z, layer_height)
        z_list = np.concatenate((z_list, np.array([z])))
    # bo = BOPAlgo_Builder()
    # bo = BOPAlgo_MakerVolume()
    bo = BOPAlgo_Splitter()
    bo.AddArgument(item)
    for i in z_list:
        p1, v1 = gp_Pnt(0, 0, i), gp_Vec(0, 0, 1)
        fc1 = make_face(
            gp_Pln(p1, vec_to_dir(v1)), -plan_len, plan_len, -plan_len, plan_len
        )
        bo.AddTool(fc1)
    if ny is not None:
        y = ymax - ymin
        y_list = np.linspace(0, y, ny)
        for i in y_list:
            p1, v1 = gp_Pnt(0, i, 0), gp_Vec(0, 1, 0)
            fc1 = make_face(
                gp_Pln(p1, vec_to_dir(v1)), -plan_len, plan_len, -plan_len, plan_len
            )
            bo.AddArgument(fc1)
    if nx is not None:
        x = xmax - xmin
        x_list = np.linspace(0, x, nx)
        for i in x_list:
            p1, v1 = gp_Pnt(i, 0, 0), gp_Vec(1, 0, 0)
            fc1 = make_face(
                gp_Pln(p1, vec_to_dir(v1)), -plan_len, plan_len, -plan_len, plan_len
            )
            bo.AddArgument(fc1)
    bo.Perform()
    top = Topo(bo.Shape())
    geo = create_compound(top.solids())
    return geo


def get_occ_bounding_box(shape: TopoDS_Shape) -> tuple:
    """
    @brief Get bounding box of occupied space of topo shape.
    @param shape TopoDS_Shape to be searched for occupied space
    @return bounding box of occupied space in x y z coordinates
    """
    bbox = Bnd_Box()
    add_bbox = brepbndlib_Add(shape, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    return xmin, ymin, zmin, xmax, ymax, zmax


def explore_topo(shape: TopoDS_Shape, shape_type: str) -> list:
    """
    @brief TopoDS Explorer for shape_type. This is a wrapper around TopExp_Explorer to allow more flexibility in the explorer
    @param shape TopoDS_Shape to be explored.
    @param shape_type Type of shape e. g. wire face shell solid compound edge
    @return List of TopoDS_Shape that are explored by shape_type. Example : [ TopoDS_Shape ( " face " ) TopoDS_Shape ( " shell "
    """
    result = []
    map_type = {
        "wire": TopAbs_WIRE,
        "face": TopAbs_FACE,
        "shell": TopAbs_SHELL,
        "solid": TopAbs_SOLID,
        "compound": TopAbs_COMPOUND,
        "edge": TopAbs_EDGE,
    }
    explorer = TopExp_Explorer(shape, map_type[shape_type])
    # This method is called by the explorer.
    while explorer.More():
        result.append(explorer.Current())
        explorer.Next()
    return result


def intersect(item: TopoDS_Shape, position: float, axis: str) -> TopoDS_Shape:
    """
    Returns the topo shape intersecting the item at the given position.

    Args:
        item: TopoDS_Shape to be intersected.
        position: Position of the plane in world coordinates.
        axis: Axis along which of the direction.
    return TopoDS_Shape with intersection or empty TopoDS_Shape if no intersection is found.
    """
    intsctr = BRepAlgoAPI_Common
    xmin, ymin, zmin, xmax, ymax, zmax = get_occ_bounding_box(item)
    bnd_x = abs(xmin - xmax) * 1.2
    bnd_y = abs(ymin - ymax) * 1.2
    bnd_z = abs(zmin - zmax) * 1.2
    match axis:
        case "z":
            plan_len = max(bnd_x, bnd_y)
            p1, v1 = gp_Pnt(0, 0, position), gp_Vec(0, 0, 1)
            fc1 = make_face(
                gp_Pln(p1, vec_to_dir(v1)), -plan_len, plan_len, -plan_len, plan_len
            )
    common = intsctr(item, fc1)
    common.Build()


def scale(item: TopoDS_Shape, cnt_pnt: gp_Pnt, factor: float) -> TopoDS_Shape:
    """
    @brief Scales TopoDS_Shape to a given value. This is useful for scaling shapes that are in a shape with respect to another shape.
    @param item TopoDS_Shape to be scaled.
    @param cnt_pnt the point of the scaling center.
    @param factor Factor to scale the shape by. Default is 1.
    @return a scaled TopoDS_Shape with scaling applied to it.
    """
    scaling_transform = gp_Trsf()
    scaling_transform.SetScale(cnt_pnt, factor)
    scaled_shape = BRepBuilderAPI_Transform(item, scaling_transform, True).Shape()
    return scaled_shape


def carve_hollow(face: TopoDS_Shape, factor: float):
    """
    @brief (This can be replaced by cutter3D() now.)Carving on a face with a shape scaling down from itself.
    @param face TopoDS_Shape to be cutted.
    @param factor Factor to be used to scale the cutter.
    @return A shape with the cutter in it's center of mass scaled by factor
    """
    cnt = get_face_center_of_mass(face, gp_pnt=True)
    cutter = scale(face, cnt, factor)
    cut = BRepAlgoAPI_Cut(face, cutter).Shape()
    return cut


def rotate_face(shape: TopoDS_Shape, angle: float, axis: str = "z", cnt: tuple = None):
    """
    @brief Rotate the topography by the given angle around the center of mass of the face.
    @param shape TopoDS_Shape to be rotated.
    @param angle Angle ( in degrees ) to rotate by.
    @param axis determine the rotation axis.
    @return the rotated shape.
    """
    transform = gp_Trsf()
    if cnt is None:
        cnt = get_face_center_of_mass(shape, gp_pnt=True)
    match axis:
        case "z":
            ax = gp_Ax1(cnt, gp_Dir(0, 0, 1))
        case "y":
            ax = gp_Ax1(cnt, gp_Dir(0, 1, 0))
        case "x":
            ax = gp_Ax1(cnt, gp_Dir(1, 0, 0))
    transform.SetRotation(ax, angle)
    transformed = BRepBuilderAPI_Transform(shape, transform).Shape()
    return transformed


def fuse(shape1: TopoDS_Shape, shape2: TopoDS_Shape) -> TopoDS_Shape:
    """
    @brief Fuse two shapes into one.
    @param shape1 first shape to fuse.
    @param shape2 second shape to fuse.
    @return topoDS_Shape
    """
    fuse = BRepAlgoAPI_Fuse(shape1, shape2).Shape()
    return fuse


def cut(shape1: TopoDS_Shape, shape2: TopoDS_Shape) -> TopoDS_Shape:
    """
    @brief Cut a TopoDS_Shape from shape1 by shape2. It is possible to use this function to cut an object in 3D
    @param shape1 shape that is to be cut
    @param shape2 shape that is to be cut. It is possible to use this function to cut an object in 3D
    @return a shape that is the result of cutting shape1 by shape2 ( or None if shape1 and shape2 are equal
    """
    comm = BRepAlgoAPI_Cut(shape1, shape2)
    return comm.Shape()


def common(shape1: TopoDS_Shape, shape2: TopoDS_Shape) -> TopoDS_Shape:
    """
    @brief Common between two TopoDS_Shapes. The result is a shape that has all components of shape1 and shape2
    @param shape1 the first shape to be compared
    @param shape2 the second shape to be compared ( must be same shape! )
    @return the common shape or None if there is no common shape between shape1 and shape2 in the sense that both shapes are
    """
    comm = BRepAlgoAPI_Common(shape1, shape2)
    return comm.Shape()


def get_boundary(item: TopoDS_Shape) -> TopoDS_Wire:
    bbox = get_occ_bounding_box(item)
    edge = explore_topo(item, "edge")  # get all edges from imported model.
    xx = []
    yy = []
    # select all edges on the boundary.
    for e in edge:
        xmin, ymin, zmin, xmax, ymax, zmax = get_occ_bounding_box(
            e
        )  # get bounding box of an edge
        if (ymin + ymax < 1e-3) or (
            abs((ymin + ymax) * 0.5 - bbox[4]) < 1e-3
        ):  # if the edge is either
            xx.append(e)
        if (xmin + xmax < 1e-3) or (abs((xmin + xmax) * 0.5 - bbox[3]) < 1e-3):
            yy.append(e)
    edges = xx + yy
    # build a compound of all edges
    wire = create_compound(edges)
    return wire


def get_face_center_of_mass(face: TopoDS_Face, gp_pnt: bool = False) -> tuple:
    """
    @brief Get the center of mass of a TopoDS_Face. This is useful for determining the center of mass of a face or to get the centre of mass of an object's surface.
    @param face TopoDS_Face to get the center of mass of
    @param gp_pnt If True return an gp_Pnt object otherwise a tuple of coordinates.
    """
    props = GProp_GProps()
    brepgprop_SurfaceProperties(face, props)
    face_surf = props.CentreOfMass()
    # face_surf. Coord if gp_pnt returns the face surface.
    if gp_pnt:
        return face_surf
    else:
        return face_surf.Coord()


def get_volume_center_of_mass(vol: TopoDS_Solid, gp_pnt: bool = False):
    """
    @brief Return the center of mass of a volume. This is an approximation of the centroid of the volume.
    @param vol TopoDS_Solid object representing a volume
    @param gp_pnt If True return an gp_Pnt object otherwise a tuple of coordinates.
    @return Center of mass of a volume.
    """
    props = GProp_GProps()
    brepgprop_VolumeProperties(vol, props)
    cog = props.CentreOfMass()
    # Return the current coordinate of the current coordinate system.
    if gp_pnt:
        return cog
    else:
        return cog.Coord()


def get_face_area(face: TopoDS_Face) -> float:
    """
    @brief Get the area of a TopoDS_Face. This is an approximation of the area of the face.
    @param face to get the area of.
    @return The area of the face.
    """
    props = GProp_GProps()
    brepgprop_SurfaceProperties(face, props)
    face_area = props.Mass()
    return face_area


def get_faces(_shape):
    """
    @brief Get faces of a shape. This is a list of topods_Face objects that correspond to the faces of the shape
    @param _shape shape to get faces of
    @return list of topods_Face objects ( one for each face in the shape ) for each face
    """
    topExp = TopExp_Explorer()
    topExp.Init(_shape, TopAbs_FACE)
    _faces = []

    # Add faces to the faces list
    while topExp.More():
        fc = topods_Face(topExp.Current())
        _faces.append(fc)
        topExp.Next()

    return _faces


def traverse(item: TopoDS_Shape) -> Topo:
    return Topo(item)


def create_polygon(points: list, isface: bool = True) -> TopoDS_Face or TopoDS_Wire:
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


def create_wire_by_points(points: list):
    """
    @brief Create a closed wire (loop) by points. The wire is defined by a list of points which are connected by an edge.
    @param points A list of points. Each point is a gp_Pnt ( x y z) where x, y and z are the coordinates of a point.
    @return A wire with the given points connected by an edge. This will be an instance of : class : `BRepBuilderAPI_MakeWire`
    """
    pts = points
    # Create a wire for each point in the list of points.
    for i, pt in enumerate(pts):
        # Create a wire for the i th point.
        if i == 0:
            edge = create_edge(pt, pts[i + 1])
            wire = create_wire(edge)
        # Create a wire for the given points.
        if i != len(pts) - 1:
            edge = create_edge(pt, pts[i + 1])
            wire = create_wire(wire, edge)
        else:
            edge = create_edge(pts[i], pts[0])
            wire = create_wire(wire, edge)
    return wire
