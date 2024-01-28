from pprint import pprint

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import OCC.Core.BRepBuilderAPI as BRepBuilderAPI
import OCC.Core.gp as gp
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BOPAlgo import BOPAlgo_Builder, BOPAlgo_MakerVolume, BOPAlgo_Splitter
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
from OCC.Core.BRepClass3d import BRepClass3d_Intersector3d
from OCC.Core.BRepGProp import brepgprop_SurfaceProperties, brepgprop_VolumeProperties
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakePrism
from OCC.Core.gp import gp_Ax1, gp_Dir, gp_Pln, gp_Pnt, gp_Trsf, gp_Vec
from OCC.Core.GProp import GProp_GProps
from OCC.Core.TopAbs import (
    TopAbs_COMPOUND,
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_FORWARD,
    TopAbs_SHELL,
    TopAbs_SOLID,
    TopAbs_WIRE,
)
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopoDS import (
    TopoDS_Compound,
    TopoDS_Face,
    TopoDS_Iterator,
    TopoDS_Shape,
    TopoDS_Shell,
    TopoDS_Solid,
    TopoDS_Wire,
    topods_Face,
)
from OCC.Core.TopTools import TopTools_ListOfShape
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCCUtils.Construct import make_face, vec_to_dir
from OCCUtils.Topology import Topo

import amworkflow.geometry.builtinCAD as bcad


def geometry_builder(*args):
    builder = BRep_Builder()
    obj = TopoDS_Compound()
    builder.MakeCompound(obj)
    for item in args[0]:
        builder.Add(obj, item)
    return obj


def sewer(*component) -> TopoDS_Shape:
    sewing = BRepBuilderAPI_Sewing()
    for i in range(len(component[0])):
        sewing.Add(component[0][i])
    sewing.Perform()
    sewed_shape = sewing.SewedShape()
    return sewed_shape


def solid_maker(item: TopoDS_Shape) -> TopoDS_Shape:
    return BRepBuilderAPI_MakeSolid(item).Shape()


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


def split(
    item: TopoDS_Shape,
    nz: int = None,
    layer_thickness: float = None,
    split_z: bool = True,
    split_x: bool = False,
    split_y: bool = False,
    nx: int = None,
    ny: int = None,
):
    """
    @brief Split a TopoDS_Shape into sub - shapes.
    @param item TopoDS_Shape to be split.
    @param nz Number of z - points to split.
    @param layer_thickness Layer thickness ( m ).
    @param split_z Split on the Z direction.
    @param split_x Split on the X direction.
    @param split_y Split on the Y direction.
    @param nx Number of sub - shapes in the x - direction.
    @param ny Number of sub - shapes in the y - direction.
    @return a compound of sub-shapes
    """
    xmin, ymin, zmin, xmax, ymax, zmax = get_occ_bounding_box(item)
    plan_len = 1.2 * max(abs(xmin - xmax), abs(ymin - ymax))
    z = zmax - zmin
    if nz != None:
        z_list = np.linspace(zmin, z, nz + 1)
    if layer_thickness != None:
        z_list = np.arange(zmin, z, layer_thickness)
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
    if ny != None:
        y = ymax - ymin
        y_list = np.linspace(0, y, ny)
        for i in y_list:
            p1, v1 = gp_Pnt(0, i, 0), gp_Vec(0, 1, 0)
            fc1 = make_face(
                gp_Pln(p1, vec_to_dir(v1)), -plan_len, plan_len, -plan_len, plan_len
            )
            bo.AddArgument(fc1)
    if nx != None:
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
    geo = geometry_builder(top.solids())
    return geo


def intersector(item: TopoDS_Shape, position: float, axis: str) -> TopoDS_Shape:
    """
    @brief Returns the topo shape intersecting the item at the given position.
    @param position Position of the plane in world coordinates.
    @param axis Axis along which of the direction.
    @return TopoDS_Shape with intersection or empty TopoDS_Shape if no intersection is found.
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

    return TopoDS_Iterator(common.Shape()).Value()


def scaler(item: TopoDS_Shape, cnt_pnt: gp_Pnt, factor: float) -> TopoDS_Shape:
    """
    @brief Scales TopoDS_Shape to a given value. This is useful for scaling shapes that are in a shape with respect to another shape.
    @param item TopoDS_Shape to be scaled.
    @param cnt_pnt the point of the scaling center.
    @param factor Factor to scale the shape by. Default is 1.
    @return a scaled TopoDS_Shape with scaling applied to it.
    """
    scaling_transform = gp.gp_Trsf()
    scaling_transform.SetScale(cnt_pnt, factor)
    scaled_shape = BRepBuilderAPI.BRepBuilderAPI_Transform(
        item, scaling_transform, True
    ).Shape()
    return scaled_shape


def hollow_carver(face: TopoDS_Shape, factor: float):
    """
    @brief (This can be replaced by cutter3D() now.)Carving on a face with a shape scaling down from itself.
    @param face TopoDS_Shape to be cutted.
    @param factor Factor to be used to scale the cutter.
    @return A shape with the cutter in it's center of mass scaled by factor
    """
    cnt = get_face_center_of_mass(face, gp_pnt=True)
    cutter = scaler(face, cnt, factor)
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


def fuser(shape1: TopoDS_Shape, shape2: TopoDS_Shape) -> TopoDS_Shape:
    """
    @brief Fuse two shapes into one.
    @param shape1 first shape to fuse.
    @param shape2 second shape to fuse.
    @return topoDS_Shape
    """
    fuse = BRepAlgoAPI_Fuse(shape1, shape2).Shape()
    return fuse


def cutter3D(shape1: TopoDS_Shape, shape2: TopoDS_Shape) -> TopoDS_Shape:
    """
    @brief Cut a TopoDS_Shape from shape1 by shape2. It is possible to use this function to cut an object in 3D
    @param shape1 shape that is to be cut
    @param shape2 shape that is to be cut. It is possible to use this function to cut an object in 3D
    @return a shape that is the result of cutting shape1 by shape2 ( or None if shape1 and shape2 are equal
    """
    comm = BRepAlgoAPI_Cut(shape1, shape2)
    return comm.Shape()


def split2(item: TopoDS_Shape, *tools: TopoDS_Shape) -> TopoDS_Compound:
    top_list = TopTools_ListOfShape()
    for i in tools:
        top_list.Append(i)
    cut = BOPAlgo_Splitter()
    print(tools)
    cut.SetArguments(top_list)
    cut.Perform()
    return cut.Shape()


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
    edge = topo_explorer(item, "edge")  # get all edges from imported model.
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
    wire = geometry_builder(edges)
    return wire


def bender(
    point_cordinates,
    radius: float = None,
    mx_pt: np.ndarray = None,
    mn_pt: np.ndarray = None,
):
    coord_t = np.array(point_cordinates).T
    if mx_pt is None:
        mx_pt = np.max(coord_t, 1)
    if mn_pt is None:
        mn_pt = np.min(coord_t, 1)
    cnt = 0.5 * (mn_pt + mx_pt)
    scale = np.abs(mn_pt - mx_pt)
    if radius is None:
        radius = scale[1] * 2
    o_y = scale[1] * 0.5 + radius
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
        pt[1] = o_y - Rp * np.cos(thetp)
    return point_cordinates


def array_project(array: np.ndarray, direct: np.ndarray) -> np.ndarray:
    """
    Project an array to the specified direction.
    """
    direct = direct / np.linalg.norm(direct)
    return np.dot(array, direct) * direct


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


def point_coord(p: gp_Pnt) -> tuple:
    """
    @brief Returns the coord of a point. This is useful for debugging and to get the coordinates of an object that is a part of a geometry.
    @param p gp_Pnt to get the coord of
    @return tuple of the coordinate of the point ( x y z ) or None if not a point ( in which case the coordinates are None
    """
    return p.Coord()


def topo_explorer(shape: TopoDS_Shape, shape_type: str) -> list:
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


def traverser(item: TopoDS_Shape) -> Topo:
    return Topo(item)


def p_bounding_box(pts: list):
    pts = np.array(pts)
    coord_t = np.array(pts).T
    mx_pt = np.max(coord_t, 1)
    mn_pt = np.min(coord_t, 1)
    return mx_pt, mn_pt


def shortest_distance_point_line(line, p):
    pt1, pt2 = line
    s = pt2 - pt1
    lmbda = (p - pt1).dot(s) / s.dot(s)
    if lmbda < 1 and lmbda > 0:
        pt_compute = pt1 + lmbda * s
        distance = np.linalg.norm(pt_compute - p)
        return lmbda, distance
    elif lmbda <= 0:
        distance = np.linalg.norm(pt1 - p)
        return 0, distance
    else:
        distance = np.linalg.norm(pt2 - p)
        return 1, distance


def shortest_distance_line_line(line1, line2):
    pt11, pt12 = line1
    pt21, pt22 = line2
    s1 = pt12 - pt11
    s2 = pt22 - pt21
    s1square = np.dot(s1, s1)
    s2square = np.dot(s2, s2)
    term1 = s1square * s2square - (np.dot(s1, s2) ** 2)
    term2 = s1square * s2square - (np.dot(s1, s2) ** 2)
    if np.isclose(term1, 0) or np.isclose(term2, 0):
        if np.isclose(s1[0], 0):
            s_p = np.array([-s1[1], s1[0], 0])
        else:
            s_p = np.array([s1[1], -s1[0], 0])
        l1 = np.random.randint(1, 4) * 0.1
        l2 = np.random.randint(6, 9) * 0.1
        pt1i = s1 * l1 + pt11
        pt2i = s2 * l2 + pt21
        si = pt2i - pt1i
        dist = np.linalg.norm(
            si * (si * s_p) / (np.linalg.norm(si) * np.linalg.norm(s_p))
        )
        return dist, np.array([pt1i, pt2i])
    lmbda1 = (
        np.dot(s1, s2) * np.dot(pt11 - pt21, s2) - s2square * np.dot(pt11 - pt21, s1)
    ) / (s1square * s2square - (np.dot(s1, s2) ** 2))
    lmbda2 = -(
        np.dot(s1, s2) * np.dot(pt11 - pt21, s1) - s1square * np.dot(pt11 - pt21, s2)
    ) / (s1square * s2square - (np.dot(s1, s2) ** 2))
    condition1 = lmbda1 >= 1
    condition2 = lmbda1 <= 0
    condition3 = lmbda2 >= 1
    condition4 = lmbda2 <= 0
    if condition1 or condition2 or condition3 or condition4:
        choices = [
            [line2, pt11, s2],
            [line2, pt12, s2],
            [line1, pt21, s1],
            [line1, pt22, s1],
        ]
        result = np.zeros((4, 2))
        for i in range(4):
            result[i] = shortest_distance_point_line(choices[i][0], choices[i][1])
        shortest_index = np.argmin(result.T[1])
        shortest_result = result[shortest_index]
        pti1 = (
            shortest_result[0] * choices[shortest_index][2]
            + choices[shortest_index][0][0]
        )
        pti2 = choices[shortest_index][1]
        # print(result)
    else:
        pti1 = pt11 + lmbda1 * s1
        pti2 = pt21 + lmbda2 * s2
    # print(lmbda1, lmbda2)
    # print(pti1, pti2)
    # print(np.dot(s1,pti2 - pti1), np.dot(s2,pti2 - pti1))
    distance = np.linalg.norm(pti1 - pti2)
    return distance, np.array([pti1, pti2])


def check_parallel_line_line(line1: np.ndarray, line2: np.ndarray) -> tuple:
    parallel = False
    colinear = False
    pt1, pt2 = line1
    pt3, pt4 = line2

    def generate_link():
        t1 = np.random.randint(1, 4) * 0.1
        t2 = np.random.randint(5, 9) * 0.1
        pt12 = (1 - t1) * pt1 + t1 * pt2
        pt34 = (1 - t2) * pt3 + t2 * pt4
        norm_L3 = np.linalg.norm(pt34 - pt12)
        while np.isclose(norm_L3, 0):
            t1 = np.random.randint(1, 4) * 0.1
            t2 = np.random.randint(5, 9) * 0.1
            pt12 = (1 - t1) * pt1 + t1 * pt2
            pt34 = (1 - t2) * pt3 + t2 * pt4
            norm_L3 = np.linalg.norm(pt34 - pt12)
        return pt12, pt34

    pt12, pt34 = generate_link()
    L1 = (pt2 - pt1) / np.linalg.norm(pt2 - pt1)
    L2 = (pt4 - pt3) / np.linalg.norm(pt4 - pt3)
    L3 = (pt34 - pt12) / np.linalg.norm(pt34 - pt12)
    if np.isclose(np.linalg.norm(np.cross(L1, L2)), 0):
        parallel = True
    if np.isclose(np.linalg.norm(np.cross(L1, L3)), 0) and parallel:
        colinear = True
    return parallel, colinear


def check_overlap(line1: np.ndarray, line2: np.ndarray) -> np.ndarray:
    A, B = line1
    C, D = line2
    s = B - A
    dist_s = np.linalg.norm(s)
    norm_s = s / dist_s
    c = C - A
    dist_c = np.linalg.norm(c)
    if np.isclose(dist_c, 0):
        lmbda_c = 0
    else:
        norm_c = c / dist_c
        sign_c = -1 if np.isclose(np.sum(norm_c + norm_s), 0) else 1
        lmbda_c = sign_c * dist_c / dist_s
    d = D - A
    dist_d = np.linalg.norm(d)
    if np.isclose(dist_d, 0):
        lmbda_d = 0
    else:
        norm_d = d / dist_d
        sign_d = -1 if np.isclose(np.sum(norm_d + norm_s), 0) else 1
        lmbda_d = sign_d * dist_d / dist_s
    indicator = np.zeros(4)
    direction_cd = lmbda_d - lmbda_c
    smaller = min(lmbda_c, lmbda_d)
    larger = max(lmbda_c, lmbda_d)
    pnt_list = np.array([A, B, C, D])
    if lmbda_c < 1 and lmbda_c > 0:
        indicator[2] = 1
    if lmbda_d < 1 and lmbda_d > 0:
        indicator[3] = 1
    if 0 < larger and 0 > smaller:
        indicator[0] = 1
    if 1 < larger and 1 > smaller:
        indicator[1] = 1
    return np.where(indicator == 1)[0], np.unique(
        pnt_list[np.where(indicator == 1)[0]], axis=0
    )


def p_get_face_area(points: list):
    pts = np.array(points).T
    x = pts[0]
    y = pts[1]
    result = 0
    for i in range(len(x)):
        if i < len(x) - 1:
            t = x[i] * y[i + 1] - x[i + 1] * y[i]
        else:
            t = x[i] * y[0] - x[0] * y[i]
        result += t
    return np.abs(result) * 0.5


def angle_of_two_arrays(a1: np.ndarray, a2: np.ndarray, rad: bool = True) -> float:
    """
    @brief Returns the angle between two vectors. This is useful for calculating the rotation angle between a vector and another vector
    @param a1 1D array of shape ( n_features )
    @param a2 2D array of shape ( n_features )
    @param rad If True the angle is in radians otherwise in degrees
    @return Angle between a1 and a2 in degrees or radians depending on rad = True or False
    """
    dot = np.dot(a1, a2)
    norm = np.linalg.norm(a1) * np.linalg.norm(a2)
    cos_value = np.round(dot / norm, 15)
    if rad:
        return np.arccos(cos_value)
    else:
        return np.rad2deg(np.arccos(cos_value))


def random_polygon_constructor(
    points: list, isface: bool = True
) -> TopoDS_Face or TopoDS_Wire:
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


def create_face(wire: TopoDS_Wire) -> TopoDS_Face:
    """
    @brief Create a BRep face from a TopoDS_Wire. This is a convenience function to use : func : ` BRepBuilderAPI_MakeFace ` and
    @param wire The wire to create a face from. Must be a TopoDS_Wire.
    @return A Face object with the properties specified
    """
    return BRepBuilderAPI_MakeFace(wire).Face()


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
    pts = np.array(
        [
            np.array(list(i.Coord())) if isinstance(i, gp_Pnt) else np.array(i)
            for i in pts
        ]
    )
    pts = [i + direct for i in pts]
    return list(pts)


def p_center_of_mass(pts: np.ndarray) -> np.ndarray:
    pts = np.array(
        [
            np.array(list(i.Coord())) if isinstance(i, gp_Pnt) else np.array(i)
            for i in pts
        ]
    )

    return np.mean(pts.T, axis=1)


def polygon_interpolater(
    plg: np.ndarray, step_len: float = None, num: int = None, isclose: bool = True
):
    def deter_dum(line: np.ndarray):
        ratio = step_len / np.linalg.norm(line[0] - line[1])
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
        elif ratio <= 0.14:
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
        line = np.array([pt, plg[i + 1]])
        if num is not None:
            p_num = num
        else:
            p_num = deter_dum(line)
        insert_p = linear_interpolate(line, p_num)
        new_plg = np.concatenate((new_plg[: pos + 1], insert_p, new_plg[pos + 1 :]))
        pos += p_num + 1
    return new_plg


def p_rotate(
    pts: np.ndarray,
    angle_x: float = 0,
    angle_y: float = 0,
    angle_z: float = 0,
    cnt: np.ndarray = None,
) -> np.ndarray:
    pts = np.array(
        [
            np.array(list(i.Coord())) if isinstance(i, gp_Pnt) else np.array(i)
            for i in pts
        ]
    )
    com = p_center_of_mass(pts)
    if cnt is None:
        cnt = np.array([0, 0, 0])
    t_vec = cnt - com
    pts += t_vec
    rot_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle_x), -np.sin(angle_x)],
            [0, np.sin(angle_x), np.cos(angle_x)],
        ]
    )
    rot_y = np.array(
        [
            [np.cos(angle_y), 0, np.sin(angle_y)],
            [0, 1, 0],
            [-np.sin(angle_y), np.cos(angle_y), 0],
        ]
    )
    rot_z = np.array(
        [
            [np.cos(angle_z), -np.sin(angle_z), 0],
            [np.sin(angle_z), np.cos(angle_z), 0],
            [0, 0, 1],
        ]
    )
    R = rot_x @ rot_y @ rot_z
    rt_pts = pts @ R
    r_pts = rt_pts - t_vec
    return r_pts


def create_face_by_plane(pln: gp_Pln, *vt: gp_Pnt) -> TopoDS_Face:
    return make_face(pln, *vt)


def linear_interpolate(pts: np.ndarray, num: int):
    for i, pt in enumerate(pts):
        if i == len(pts) - 1:
            break
        else:
            interpolated_points = np.linspace(pt, pts[i + 1], num=num + 2)[1:-1]
    return interpolated_points


def random_pnt_gen(xmin, xmax, ymin, ymax, zmin=0, zmax=0):
    random_x = np.random.randint(xmin, xmax)
    random_y = np.random.randint(ymin, ymax)
    if zmin == 0 and zmax == 0:
        random_z = 0
    else:
        random_z = np.random.randint(zmin, zmax)
    return np.array([random_x, random_y, random_z])


def random_line_gen(xmin, xmax, ymin, ymax, zmin=0, zmax=0):
    pt1 = random_pnt_gen(xmin, xmax, ymin, ymax, zmin, zmax)
    pt2 = random_pnt_gen(xmin, xmax, ymin, ymax, zmin, zmax)
    return np.array([pt1, pt2])


class Pnt:
    def __init__(self, *coords: list):
        self.coords = coords
        if (len(self.coords) == 1) and (
            type(self.coords[0]) is list or isinstance(self.coords[0], np.ndarray)
        ):
            self.coords = coords[0]
        self.coords = self.format_coords()
        self.pts_index = {}
        self.pts_digraph = {}
        self.count_pt_id = 0
        self.init_pts_sequence = []
        self.init_pnts()

    def enclose(self):
        distance = np.linalg.norm(self.coords_numpy[-1] - self.coords_numpy[0])
        if np.isclose(distance, 0):
            print("Polygon seems already enclosed, skipping...")
        else:
            self.coords_numpy = np.vstack((self.coords_numpy, self.coords_numpy[0]))
            self.create_attr()

    # def create(self):
    #     self.coords_numpy = self.create_pnts()
    #     self.eliminate_overlap()
    #     self.create_attr()
    def format_coords(self):
        return [self.pnt(i) for i in self.coords]

    def create_attr(self):
        self.coords_to_list = self.coords_numpy.tolist()
        self.coords_to_gp_Pnt: list
        self.x, self.y, self.z = self.coords_numpy.T
        self.pts_num = self.coords_numpy.shape[0]

    def pnt(self, pt_coord) -> np.ndarray:
        opt = np.array(pt_coord)
        dim = len(pt_coord)
        if dim > 3:
            raise Exception(
                f"Got wrong point {pt_coord}: Dimension more than 3rd provided."
            )
        if dim < 3:
            opt = np.lib.pad(opt, ((0, 3 - dim)), "constant", constant_values=0)
        return opt

    def new_pnt(self, pt_coords: list):
        pt_coords = self.pnt(pt_coords)
        for i, v in self.pts_index.items():
            if self.pnt_overlap(v, pt_coords):
                return False, i
        return True, None

    def pnt_overlap(self, pt1: np.ndarray, pt2: np.ndarray) -> bool:
        return np.isclose(np.linalg.norm(pt1 - pt2), 0)

    def init_pnts(self) -> None:
        for i, pt in enumerate(self.coords):
            pt_id = self.register_pnt(pt)
            if i != len(self.coords) - 1:
                self.init_pts_sequence.append(pt_id)
            if i != 0:
                self.update_digraph(self.init_pts_sequence[i - 1], pt_id)
                self.init_pts_sequence[i - 1] = [self.init_pts_sequence[i - 1], pt_id]

    def register_pnt(self, pt: list) -> int:
        pnt = self.pnt(pt)
        new, old_id = self.new_pnt(pnt)
        if new:
            self.pts_index.update({self.count_pt_id: pnt})
            pnt_id = self.count_pt_id
            self.count_pt_id += 1
            return pnt_id
        else:
            return old_id

    def update_digraph(
        self,
        start_node: int,
        end_node: int,
        insert_node: int = None,
        build_new_edge: bool = True,
    ) -> None:
        if start_node not in self.pts_index:
            raise Exception(f"Unrecognized start node: {start_node}.")
        if end_node not in self.pts_index:
            raise Exception(f"Unrecognized end node: {end_node}.")
        if (insert_node not in self.pts_index) and (insert_node is not None):
            raise Exception(f"Unrecognized inserting node: {insert_node}.")
        if start_node in self.pts_digraph:
            if insert_node is None:
                self.pts_digraph[start_node].append(end_node)
            else:
                end_node_list_index = self.pts_digraph[start_node].index(end_node)
                self.pts_digraph[start_node][end_node_list_index] = insert_node
                if build_new_edge:
                    self.pts_digraph.update({insert_node: [end_node]})
        else:
            if insert_node is None:
                self.pts_digraph.update({start_node: [end_node]})
            else:
                raise Exception("No edge found for insertion option.")


class Segments(Pnt):
    def __init__(self, *coords: list):
        super().__init__(*coords)
        self.segments_index = {}
        self.modify_edge_list = {}
        self.virtual_vector = {}
        self.virtual_pnt = {}
        self.find_overlap_node_on_edge()
        self.modify_edge()

    def enclose(self):
        pass

    def get_segment(self, pt1: int, pt2: int) -> np.ndarray:
        return np.array([self.pts_index[pt1], self.pts_index[pt2]])

        # def init_segments(self):
        for i, v in enumerate(self.pts_sequance):
            if i != len(self.pts_sequance) - 1:
                self.segments_index.update({v: [self.pts_sequance[i + 1]]})
                self.count_vector_id += 1

    def insert_item(
        self, *items: np.ndarray, original: np.ndarray, insert_after: int
    ) -> np.ndarray:
        print(original[: insert_after + 1])
        return np.concatenate(
            (original[: insert_after + 1], items, original[insert_after + 1 :])
        )

    def add_pending_change(self, edge: tuple, new_node: int) -> None:
        if edge in self.modify_edge_list:
            self.modify_edge_list[edge].append(new_node)
        else:
            self.modify_edge_list.update({edge: [new_node]})

    def modify_edge(self):
        for edge, nodes in self.modify_edge_list.items():
            edge_0_coords = self.pts_index[edge[0]]
            nodes_coords = [self.pts_index[i] for i in nodes]
            distances = [np.linalg.norm(i - edge_0_coords) for i in nodes_coords]
            order = np.argsort(distances)
            nodes = [nodes[i] for i in order]
            self.pts_digraph[edge[0]].remove(edge[1])
            pts_list = [edge[0]] + nodes + [edge[1]]
            for i, nd in enumerate(pts_list):
                if i == 0:
                    continue
                self.update_digraph(pts_list[i - 1], nd, build_new_edge=False)
                if (i != 1) and (i != len(pts_list) - 1):
                    if (pts_list[i - 1] in self.virtual_pnt) and nd in (
                        self.virtual_pnt
                    ):
                        self.virtual_vector.update({(pts_list[i - 1], nd): True})

    def check_self_edge(self, line: np.ndarray) -> bool:
        if self.pnt_overlap(line[0], line[1]):
            return True
        else:
            return False

    def overlap_node_on_edge_finder(self, i, j):
        v = self.init_pts_sequence[i]
        vv = self.init_pts_sequence[j]
        print(i, j)
        lin1 = self.get_segment(v[0], v[1])
        lin2 = self.get_segment(vv[0], vv[1])
        self_edge = self.check_self_edge(lin1) or self.check_self_edge(lin2)
        if not self_edge:
            parallel, colinear = check_parallel_line_line(lin1, lin2)
            # if v == [13,14]:
            #     print("line:",(v,vv), parallel, colinear)
            if parallel:
                if colinear:
                    index, coords = check_overlap(lin1, lin2)
                    if len(index) < 4:
                        for ind in index:
                            if ind in [0, 1]:
                                self.add_pending_change(tuple(vv), v[ind])
                            else:
                                self.add_pending_change(tuple(v), vv[ind])
            else:
                distance = shortest_distance_line_line(lin1, lin2)
                intersect = np.isclose(distance[0], 0)
                new, pt = self.new_pnt(distance[1][0])
                if intersect and new:
                    pnt_id = self.register_pnt(distance[1][0])
                    self.add_pending_change(tuple(v), pnt_id)
                    self.add_pending_change(tuple(vv), pnt_id)

    def arg_generator(self):
        iter_range = range(len(self.init_pts_sequence))
        visited = {}
        for i in iter_range:
            for j in iter_range:
                if i == j:
                    continue
                if i == j + 1:
                    continue
                if j == i - 1:
                    continue
                if (i, j) in visited or (j, i) in visited:
                    continue
                args = (self, i, j)
                print(args)
                yield args
            visited.update({(i, j): True, (j, i): True})

    def find_overlap_node_on_edge(self):
        visited = {}
        for i, v in enumerate(self.init_pts_sequence):
            for j, vv in enumerate(self.init_pts_sequence):
                if i == j:
                    continue
                if i == j + 1:
                    continue
                if j == i + 1:
                    continue
                if i == len(self.init_pts_sequence) * 2 - 1 - i:
                    continue
                if (i, j) in visited or (j, i) in visited:
                    continue
                print(i, j)
                lin1 = self.get_segment(v[0], v[1])
                lin2 = self.get_segment(vv[0], vv[1])
                self_edge = self.check_self_edge(lin1) or self.check_self_edge(lin2)
                if not self_edge:
                    parallel, colinear = check_parallel_line_line(lin1, lin2)
                    # if v == [13,14]:
                    #     print("line:",(v,vv), parallel, colinear)
                    if parallel:
                        if colinear:
                            index, coords = check_overlap(lin1, lin2)
                            if len(index) < 4:
                                for ind in index:
                                    if ind in [0, 1]:
                                        self.add_pending_change(tuple(vv), v[ind])
                                    else:
                                        self.add_pending_change(tuple(v), vv[ind - 2])
                    else:
                        distance = shortest_distance_line_line(lin1, lin2)
                        intersect = np.isclose(distance[0], 0)
                        new, pt = self.new_pnt(distance[1][0])
                        if intersect and new:
                            pnt_id = self.register_pnt(distance[1][0])
                            self.virtual_pnt.update({pnt_id: True})
                            self.add_pending_change(tuple(v), pnt_id)
                            self.add_pending_change(tuple(vv), pnt_id)
                visited.update({(i, j): True, (j, i): True})


class CreateWallByPointsUpdate:
    def __init__(self, coords: list, th: float, height: float, is_close: bool = True):
        self.coords = Pnt(coords).coords
        self.height = height
        self.R = None
        self.interpolate = 8
        self.th = th
        self.is_close = is_close
        self.in_wall_pts_list = {}
        self.in_loop_pts_list = {}
        self.result_loops = []
        self.vecs = []
        self.central_segments = []
        self.dir_vecs = []
        self.ths = []
        self.lft_coords = []
        self.rgt_coords = []
        self.volume = 0
        self.side_coords: list
        self.create_sides()
        self.pnts = Segments(self.side_coords)
        self.G = nx.from_dict_of_lists(self.pnts.pts_digraph, create_using=nx.DiGraph)
        # self.all_loops = list(nx.simple_cycles(self.H)) # Dangerous! Ran out of memory.
        self.loop_generator = nx.simple_cycles(self.G)
        # self.check_pnt_in_wall()
        self.postprocessing()

    def create_sides(self):
        if self.R is not None:
            self.coords = polygon_interpolater(self.coords, self.interpolate)
            self.coords = bender(self.coords, self.R)
            self.coords = [i for i in self.coords]
        self.th *= 0.5
        for i, p in enumerate(self.coords):
            if i != len(self.coords) - 1:
                self.central_segments.append([self.coords[i], self.coords[i + 1]])
                a1 = self.coords[i + 1] - self.coords[i]
                if i == 0:
                    if self.is_close:
                        dr = angular_bisector(self.coords[-1] - p, a1)
                        # ang = angle_of_two_arrays(dir_vecs[i-1],dr)
                        ang2 = angle_of_two_arrays(
                            laterality_indicator(p - self.coords[-1], True), dr
                        )
                        ang_th = ang2
                        if ang2 > np.pi / 2:
                            dr *= -1
                            ang_th = np.pi - ang2
                        nth = np.abs(self.th / np.cos(ang_th))
                    else:
                        dr = laterality_indicator(a1, True)
                        nth = self.th
                else:
                    dr = angular_bisector(-self.vecs[i - 1], a1)
                    ang2 = angle_of_two_arrays(
                        laterality_indicator(self.vecs[i - 1], True), dr
                    )
                    ang_th = ang2
                    if ang2 > np.pi / 2:
                        dr *= -1
                        ang_th = np.pi - ang2
                    nth = np.abs(self.th / np.cos(ang_th))
            else:
                if self.is_close:
                    self.central_segments.append([self.coords[i], self.coords[0]])
                    a1 = self.coords[0] - self.coords[i]
                    dr = angular_bisector(-self.vecs[i - 1], a1)
                    ang2 = angle_of_two_arrays(
                        laterality_indicator(self.vecs[i - 1], True), dr
                    )
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
            self.side_coords = self.lft_coords + self.rgt_coords
        else:
            self.rgt_coords = self.rgt_coords[::-1]
            self.side_coords = self.lft_coords + self.rgt_coords + [self.lft_coords[0]]

    def check_pnt_in_wall(self):
        for pnt, coord in self.pnts.pts_index.items():
            for vec in self.central_segments:
                lmbda, dist = shortest_distance_point_line(vec, coord)
                if dist < 0.9 * self.th:
                    print(f"pnt:{pnt},dist:{dist},lmbda:{lmbda}, vec:{vec}")
                    self.in_wall_pts_list.update({pnt: True})
                    break

    def visualize(
        self,
        display_polygon: bool = True,
        display_central_path: bool = False,
        all_polygons: bool = False,
    ):
        # Extract the x and y coordinates and IDs
        a = self.pnts.pts_index
        x = [coord[0] for coord in a.values()]
        y = [coord[1] for coord in a.values()]
        ids = list(a.keys())  # Get the point IDs
        # Create a scatter plot in 2D
        plt.subplot(1, 2, 1)
        # plt.figure()
        plt.scatter(x, y)

        # Annotate points with IDs
        for i, (xi, yi) in enumerate(zip(x, y)):
            plt.annotate(f"{ids[i]}", (xi, yi), fontsize=12, ha="right")

        if display_polygon:
            if all_polygons:
                display_loops = nx.simple_cycles(self.G)
            else:
                display_loops = self.result_loops
            for lp in display_loops:
                coords = [self.pnts.pts_index[i] for i in lp]
                x = [point[0] for point in coords]
                y = [point[1] for point in coords]
                plt.plot(x + [x[0]], y + [y[0]], linestyle="-", marker="o")
        if display_central_path:
            talist = np.array(self.coords).T
            x1 = talist[0]
            y1 = talist[1]
            plt.plot(x1, y1, "bo-", label="central path", color="b")

        # Create segments by connecting consecutive points

        # a_subtitute = np.array(self.side_coords)
        # toutput1 = a_subtitute.T
        # x2 = toutput1[0]
        # y2 = toutput1[1]
        # plt.plot(x2, y2, 'ro-', label='outer line')

        # Set labels and title
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.title("Points With Polygons detected")

        plt.subplot(1, 2, 2)
        # layout = nx.spring_layout(self.G)
        layout = nx.circular_layout(self.G)
        # Draw the nodes and edges
        nx.draw(
            self.G,
            pos=layout,
            with_labels=True,
            node_color="skyblue",
            font_size=10,
            node_size=300,
        )
        plt.title("Multi-Digraph")
        plt.tight_layout()
        # Show the plot
        plt.show()

    def get_loops(self):
        return [i for i in self.all_loops if len(i) > 2]

    def visualize_graph(self):
        layout = nx.spring_layout(self.G)
        # Draw the nodes and edges
        nx.draw(
            self.G,
            pos=layout,
            with_labels=True,
            node_color="skyblue",
            font_size=10,
            node_size=500,
        )
        plt.title("NetworkX Graph Visualization")
        plt.show()

    def Shape(self):
        loop_r = self.rank_result_loops()
        print(loop_r)
        boundary = [self.occ_pnt(self.pnts.pts_index[i]) for i in loop_r[0]]
        poly0 = random_polygon_constructor(boundary)
        poly_r = poly0
        for i, h in enumerate(loop_r):
            if i == 0:
                continue
            h = [self.occ_pnt(self.pnts.pts_index[i]) for i in h]
            poly_c = random_polygon_constructor(h)
            poly_r = cutter3D(poly_r, poly_c)
        self.poly = poly_r
        if not np.isclose(self.height, 0):
            wall_compound = create_prism(self.poly, [0, 0, self.height])
            faces = topo_explorer(wall_compound, "face")
            wall_shell = sewer(faces)
            self.wall = solid_maker(wall_shell)
            return self.wall
        else:
            return self.poly

    def postprocessing(self):
        correct_loop_count = 0
        for lp in self.loop_generator:
            real_loop = True
            visible_loop = True
            in_wall_pt_count = 0
            virtual_vector_count = 0
            if len(lp) > 2:
                for i, pt in enumerate(lp):
                    if i == 0:
                        if pt in self.in_wall_pts_list:
                            in_wall_pt_count += 1
                        if (lp[-1], pt) in self.pnts.virtual_vector:
                            virtual_vector_count += 1
                    else:
                        if pt in self.in_wall_pts_list:
                            in_wall_pt_count += 1
                        if (lp[i - 1], pt) in self.pnts.virtual_vector:
                            virtual_vector_count += 1
                    if (
                        (in_wall_pt_count > 0)
                        or (virtual_vector_count > 1)
                        # or ((in_wall_pt_count == 0) and (virtual_vector_count > 0))
                    ):
                        # if (in_wall_pt_count > 0):
                        visible_loop = False
                        break
            else:
                real_loop = False
            if real_loop and visible_loop:
                self.result_loops.append(lp)
                for pt in lp:
                    if pt not in self.in_loop_pts_list:
                        self.in_loop_pts_list.update({pt: [correct_loop_count]})
                    else:
                        self.in_loop_pts_list[pt].append(correct_loop_count)
                correct_loop_count += 1
        loop_counter = np.zeros(len(self.result_loops))
        visited = []
        counter = 0
        for pt, lp in self.in_loop_pts_list.items():
            if len(lp) > 1:
                if counter == 0:
                    visited.append(lp)
                    for ind in lp:
                        loop_counter[ind] += 1
                else:
                    if any(sorted(lp) == sorted(item) for item in visited):
                        continue
                    else:
                        visited.append(lp)
                        for ind in lp:
                            loop_counter[ind] += 1
                counter += 1
        filtered_lp = np.where(loop_counter > 1)[0].tolist()
        print("filtered:", filtered_lp)
        print("vote:", loop_counter)
        self.result_loops = [
            v for i, v in enumerate(self.result_loops) if i not in filtered_lp
        ]
        print("result:", self.result_loops)

    def rank_result_loops(self):
        areas = np.zeros(len(self.result_loops))
        for i, lp in enumerate(self.result_loops):
            lp_coord = [self.pnts.pts_index[i] for i in lp]
            area = p_get_face_area(lp_coord)
            areas[i] = area
        rank = np.argsort(areas).tolist()
        self.volume = (2 * np.max(areas) - np.sum(areas)) * self.height * 1e-6
        self.result_loops = sorted(
            self.result_loops,
            key=lambda x: rank.index(self.result_loops.index(x)),
            reverse=True,
        )
        return self.result_loops

    def occ_pnt(self, coord) -> gp_Pnt:
        return gp_Pnt(*coord)
