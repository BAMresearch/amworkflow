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
    Create a box with given length, width, height, and radius. If radius is None or 0, the box will be sewed by a solid.

    :param float length: Length of the box.
    :param float width: Width of the box.
    :param float height: Height of the box.
    :param float radius: Radius of the box. Default is None, which means that the box is without curves.
    :param float alpha: Angle for bending the box. Default is half the length divided by the radius.
    :param bool shell: If True, the box will be a shell. Default is False.
    :return: The created box in TopoDS_Shape form.
    :rtype: TopoDS_Shape
    """
    if (radius is None) or (radius == 0):
        if shell:
            box = BRepPrimAPI_MakeBox(length, width, height).Shape()
            faces = list(Topo(TopoDS_Solid).faces_from_solids(box))
            sewed_face = sew_face(faces)
            return sewed_face
        else:
            return BRepPrimAPI_MakeBox(length, width, height).Shape()
    else:
        if alpha is None:
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
        for i in range(len(component)):
            sewing.Add(component[i])
        sewing.Perform()
        sewed_shape = sewing.SewedShape()
        solid = BRepBuilderAPI_MakeSolid(sewed_shape).Shape()
        curve_box = create_compound(component)
        if shell:
            return sewed_shape
        else:
            return solid


def create_prism(shape: TopoDS_Shape, vector: list, copy: bool = True) -> TopoDS_Shell:
    """
    Create a prism from a TopoDS_Shape and vector. 

    :param TopoDS_Shape shape: The shape to be used as the base.
    :param list vector: A list of 3 elements (x, y, z). Normally only z is used to define the height of the prism.
    :param bool copy: If True, the base wire(s) will be copied. Recommended to always use True.
    :return: The created prism.
    :rtype: TopoDS_Shell
    """
    return BRepPrimAPI_MakePrism(
        shape, gp_Vec(vector[0], vector[1], vector[2]), copy
    ).Shape()


def create_wire(*edge) -> TopoDS_Wire:
    """
    Create a wire from the given edge(s).

    :param edge: One or more edges to build a wire.
    :return: A wire built from the given edge(s).
    :rtype: TopoDS_Wire
    """
    return BRepBuilderAPI_MakeWire(*edge).Wire()


def create_face(wire: TopoDS_Wire) -> TopoDS_Face:
    """
    Create a BRep face from a TopoDS_Wire.

    :param TopoDS_Wire wire: The wire to create a face from.
    :return: A face created from the given wire.
    :rtype: TopoDS_Face
    """
    return BRepBuilderAPI_MakeFace(wire).Face()


def create_edge(
    pnt1: gp_Pnt = None, pnt2: gp_Pnt = None, arch: Geom_TrimmedCurve = None
) -> TopoDS_Edge:
    """
    Create an edge between two points or from an arc.

    :param gp_Pnt pnt1: First point of the edge.
    :param gp_Pnt pnt2: Second point of the edge.
    :param Geom_TrimmedCurve arch: Arc curve to create the edge from. If None, the edge will be created from pnt1 and pnt2.
    :return: The created edge.
    :rtype: TopoDS_Edge
    """
    if isinstance(pnt1, gp_Pnt) and isinstance(pnt2, gp_Pnt):
        edge = BRepBuilderAPI_MakeEdge(pnt1, pnt2).Edge()
    elif isinstance(arch, Geom_TrimmedCurve):
        edge = BRepBuilderAPI_MakeEdge(arch).Edge()
    return edge


def sew_face(*component) -> TopoDS_Shape:
    """
    Sew multiple faces into a single shape.

    :param component: A list of faces to sew.
    :return: The sewed shape.
    :rtype: TopoDS_Shape
    """
    sewing = BRepBuilderAPI_Sewing()
    for i in range(len(component[0])):
        sewing.Add(component[0][i])
    sewing.Perform()
    sewed_shape = sewing.SewedShape()
    return sewed_shape


def create_solid(item: TopoDS_Shape) -> TopoDS_Shape:
    """
    Create a solid from a TopoDS_Shape.

    :param TopoDS_Shape item: The shape to convert to a solid.
    :return: The created solid.
    :rtype: TopoDS_Shape
    """
    return BRepBuilderAPI_MakeSolid(item).Shape()


def create_compound(*args) -> TopoDS_Compound:
    """
    Create a compound from multiple shapes.

    :param args: The shapes to combine into a compound.
    :return: The created compound.
    :rtype: TopoDS_Compound
    """
    builder = BRep_Builder()
    obj = TopoDS_Compound()
    builder.MakeCompound(obj)
    for item in args[0]:
        builder.Add(obj, item)
    return obj


def translate(item: TopoDS_Shape, vector: list):
    """
    Translate a shape by a given vector.

    :param TopoDS_Shape item: The shape to translate.
    :param list vector: A list of 3 elements [x, y, z] representing the translation vector.
    """
    ts_handler = gp_Trsf()
    ts_handler.SetTranslation(gp_Vec(vector[0], vector[1], vector[2]))
    loc = TopLoc_Location(ts_handler)
    item.Move(loc)


def reverse(item: TopoDS_Shape) -> TopoDS_Shape:
    """
    Reverse a shape.

    :param TopoDS_Shape item: The shape to reverse.
    :return: The reversed shape.
    :rtype: TopoDS_Shape
    """
    return item.Reversed()


def geom_copy(item: TopoDS_Shape) -> TopoDS_Shape:
    """
    Copy a shape.

    :param TopoDS_Shape item: The shape to copy.
    :return: The copied shape.
    :rtype: TopoDS_Shape
    """
    wire_top_builder = BRepBuilderAPI_Copy(item)
    wire_top_builder.Perform(item, True)
    new_item = wire_top_builder.Shape()
    return new_item


def split(item: TopoDS_Shape, *tools: TopoDS_Shape) -> TopoDS_Compound:
    """
    Split a shape using one or more tools.

    :param TopoDS_Shape item: The shape to split.
    :param TopoDS_Shape tools: The tools to split the shape with.
    :return: The resulting compound shape after splitting.
    :rtype: TopoDS_Compound
    """
    top_list = TopTools_ListOfShape()
    for i in tools:
        top_list.Append(i)
    cut = BOPAlgo_Splitter()
    cut.SetArguments(top_list)
    cut.Perform()
    return cut.Shape()


def split_by_plane(
    item: TopoDS_Shape,
    nz: int = None,
    layer_height: float = None,
    nx: int = None,
    ny: int = None,
) -> TopoDS_Compound:
    """
    Split a shape into sub-shapes by planes.

    :param TopoDS_Shape item: The shape to split.
    :param int nz: Number of layers in z-direction to split.
    :param float layer_height: Height of each layer.
    :param int nx: Number of sub-shapes in the x-direction.
    :param int ny: Number of sub-shapes in the y-direction.
    :return: A compound of sub-shapes.
    :rtype: TopoDS_Compound
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
    Get the bounding box of a shape.

    :param TopoDS_Shape shape: The shape to get the bounding box for.
    :return: A tuple containing (xmin, ymin, zmin, xmax, ymax, zmax).
    :rtype: tuple
    """
    bbox = Bnd_Box()
    add_bbox = brepbndlib_Add(shape, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    return xmin, ymin, zmin, xmax, ymax, zmax


def explore_topo(shape: TopoDS_Shape, shape_type: str) -> list:
    """
    Explore a shape and return a list of sub-shapes of a specified type.

    :param TopoDS_Shape shape: The shape to explore.
    :param str shape_type: The type of sub-shapes to retrieve (e.g., "wire", "face", "shell", "solid", "compound", "edge").
    :return: A list of sub-shapes.
    :rtype: list
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
    Intersect a shape with a plane at a given position.

    :param TopoDS_Shape item: The shape to intersect.
    :param float position: The position of the plane in world coordinates.
    :param str axis: The axis along which the intersection is made.
    :return: The intersected shape.
    :rtype: TopoDS_Shape
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
    Scale a shape by a given factor.

    :param TopoDS_Shape item: The shape to scale.
    :param gp_Pnt cnt_pnt: The center point for scaling.
    :param float factor: The scaling factor.
    :return: The scaled shape.
    :rtype: TopoDS_Shape
    """
    scaling_transform = gp_Trsf()
    scaling_transform.SetScale(cnt_pnt, factor)
    scaled_shape = BRepBuilderAPI_Transform(item, scaling_transform, True).Shape()
    return scaled_shape


def carve_hollow(face: TopoDS_Shape, factor: float) -> TopoDS_Shape:
    """
    Carve a hollow in a face by scaling it down from itself.

    :param TopoDS_Shape face: The face to carve.
    :param float factor: The scaling factor.
    :return: The carved shape.
    :rtype: TopoDS_Shape
    """
    cnt = get_face_center_of_mass(face, gp_pnt=True)
    cutter = scale(face, cnt, factor)
    cut = BRepAlgoAPI_Cut(face, cutter).Shape()
    return cut


def rotate_face(
    shape: TopoDS_Shape, angle: float, axis: str = "z", cnt: tuple = None
) -> TopoDS_Shape:
    """
    Rotate a shape around its center of mass by a given angle.

    :param TopoDS_Shape shape: The shape to rotate.
    :param float angle: The angle to rotate by (in degrees).
    :param str axis: The axis to rotate around (default is "z").
    :param tuple cnt: The center of rotation. If None, the center of mass is used.
    :return: The rotated shape.
    :rtype: TopoDS_Shape
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
    Fuse two shapes into one.

    :param TopoDS_Shape shape1: The first shape to fuse.
    :param TopoDS_Shape shape2: The second shape to fuse.
    :return: The fused shape.
    :rtype: TopoDS_Shape
    """
    fuse = BRepAlgoAPI_Fuse(shape1, shape2).Shape()
    return fuse


def cut(shape1: TopoDS_Shape, shape2: TopoDS_Shape) -> TopoDS_Shape:
    """
    Cut one shape from another.

    :param TopoDS_Shape shape1: The shape to cut from.
    :param TopoDS_Shape shape2: The shape to cut with.
    :return: The cut shape.
    :rtype: TopoDS_Shape
    """
    comm = BRepAlgoAPI_Cut(shape1, shape2)
    return comm.Shape()


def common(shape1: TopoDS_Shape, shape2: TopoDS_Shape) -> TopoDS_Shape:
    """
    Find the common volume between two shapes.

    :param TopoDS_Shape shape1: The first shape.
    :param TopoDS_Shape shape2: The second shape.
    :return: The common shape.
    :rtype: TopoDS_Shape
    """
    comm = BRepAlgoAPI_Common(shape1, shape2)
    return comm.Shape()


def get_boundary(item: TopoDS_Shape) -> TopoDS_Wire:
    """
    Get the boundary edges of a shape.

    :param TopoDS_Shape item: The shape to get the boundary for.
    :return: The boundary wire.
    :rtype: TopoDS_Wire
    """
    bbox = get_occ_bounding_box(item)
    edge = explore_topo(item, "edge")
    xx = []
    yy = []
    for e in edge:
        xmin, ymin, zmin, xmax, ymax, zmax = get_occ_bounding_box(e)
        if (ymin + ymax < 1e-3) or (
            abs((ymin + ymax) * 0.5 - bbox[4]) < 1e-3
        ):
            xx.append(e)
        if (xmin + xmax < 1e-3) or (abs((xmin + xmax) * 0.5 - bbox[3]) < 1e-3):
            yy.append(e)
    edges = xx + yy
    wire = create_compound(edges)
    return wire


def get_face_center_of_mass(face: TopoDS_Face, gp_pnt: bool = False):
    """
    Get the center of mass of a face.

    :param TopoDS_Face face: The face to get the center of mass for.
    :param bool gp_pnt: If True, return a gp_Pnt object; otherwise, return a tuple of coordinates.
    :return: The center of mass.
    :rtype: gp_Pnt or tuple
    """
    props = GProp_GProps()
    brepgprop_SurfaceProperties(face, props)
    face_surf = props.CentreOfMass()
    if gp_pnt:
        return face_surf
    else:
        return face_surf.Coord()


def get_volume_center_of_mass(vol: TopoDS_Solid, gp_pnt: bool = False):
    """
    Get the center of mass of a volume.

    :param TopoDS_Solid vol: The volume to get the center of mass for.
    :param bool gp_pnt: If True, return a gp_Pnt object; otherwise, return a tuple of coordinates.
    :return: The center of mass.
    :rtype: gp_Pnt or tuple
    """
    props = GProp_GProps()
    brepgprop_VolumeProperties(vol, props)
    cog = props.CentreOfMass()
    if gp_pnt:
        return cog
    else:
        return cog.Coord()


def get_face_area(face: TopoDS_Face) -> float:
    """
    Get the area of a face.

    :param TopoDS_Face face: The face to get the area for.
    :return: The area of the face.
    :rtype: float
    """
    props = GProp_GProps()
    brepgprop_SurfaceProperties(face, props)
    face_area = props.Mass()
    return face_area


def get_faces(_shape):
    """
    Get the faces of a shape.

    :param TopoDS_Shape _shape: The shape to get the faces for.
    :return: A list of faces.
    :rtype: list
    """
    topExp = TopExp_Explorer()
    topExp.Init(_shape, TopAbs_FACE)
    _faces = []
    while topExp.More():
        fc = topods_Face(topExp.Current())
        _faces.append(fc)
        topExp.Next()
    return _faces


def traverse(item: TopoDS_Shape) -> Topo:
    """
    Traverse a shape using Topo.

    :param TopoDS_Shape item: The shape to traverse.
    :return: A Topo object representing the traversed shape.
    :rtype: Topo
    """
    return Topo(item)


def create_polygon(points: list, isface: bool = True) -> TopoDS_Face or TopoDS_Wire:
    """
    Create a polygon from a list of points.

    :param list points: The list of points defining the polygon.
    :param bool isface: If True, the polygon will be face-oriented; otherwise, it will be a wire.
    :return: The created polygon.
    :rtype: TopoDS_Face or TopoDS_Wire
    """
    pb = BRepBuilderAPI_MakePolygon()
    for pt in points:
        pb.Add(pt)
    pb.Build()
    pb.Close()
    if isface:
        return create_face(pb.Wire())
    else:
        return pb.Wire()


def create_wire_by_points(points: list) -> TopoDS_Wire:
    """
    Create a closed wire (loop) from a list of points.

    :param list points: The list of points defining the wire.
    :return: The created wire.
    :rtype: TopoDS_Wire
    """
    pts = points
    for i, pt in enumerate(pts):
        if i == 0:
            edge = create_edge(pt, pts[i + 1])
            wire = create_wire(edge)
        if i != len(pts) - 1:
            edge = create_edge(pt, pts[i + 1])
            wire = create_wire(wire, edge)
        else:
            edge = create_edge(pts[i], pts[0])
            wire = create_wire(wire, edge)
    return wire
