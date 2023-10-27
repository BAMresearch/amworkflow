import numpy as np
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BOPAlgo import BOPAlgo_Splitter
from OCC.Core.BRep import BRep_Builder
from OCC.Core.BRepBndLib import brepbndlib_Add
from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_Copy,
    BRepBuilderAPI_MakeSolid,
    BRepBuilderAPI_Sewing,
)
from OCC.Core.BRepGProp import brepgprop_SurfaceProperties
from OCC.Core.gp import gp_Pln, gp_Pnt, gp_Trsf, gp_Vec
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
from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Face, TopoDS_Shape
from OCCUtils.Construct import make_face, vec_to_dir
from OCCUtils.Topology import Topo


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


def split(
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
    assert [nz, layer_height].count(None) == 1

    xmin, ymin, zmin, xmax, ymax, zmax = get_occ_bounding_box(item)
    plan_len = 1.2 * max(abs(xmin - xmax), abs(ymin - ymax))
    z = zmax - zmin
    if nz is not None:
        z_list = np.linspace(zmin, z, nz + 1)
    if layer_height is not None:
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
