from OCC.Core.gp import gp_Pnt, gp_Trsf, gp_Vec
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Copy
from OCC.Core.TopoDS import TopoDS_Shape
import numpy as np
import OCC.Core.BRepBuilderAPI as BRepBuilderAPI
import OCC.Core.gp as gp
from OCC.Core.gp import gp_Pln
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Ax1, gp_Dir
from OCC.Core.BOPAlgo import BOPAlgo_Builder
from OCCUtils.Topology  import Topo
from OCCUtils.Construct import make_face
from OCCUtils.Construct import vec_to_dir
from amworkflow.src.geometries.builder import geometry_builder
from OCC.Core.BRepBndLib import brepbndlib_Add
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepClass3d import BRepClass3d_Intersector3d
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common
from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Iterator
from amworkflow.src.geometries.property import get_face_center_of_mass
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform, BRepBuilderAPI_Sewing
from amworkflow.src.geometries.property import get_occ_bounding_box

def translate(item: TopoDS_Shape,
                vector: list):
    """
    @brief Translates the shape by the distance and direction of a given vector.
    @param item The item to be translated. It must be a TopoDS_Shape
    @param vector The vector to translate the object by. The vector has to be a list with three elements
    """
    ts_handler = gp_Trsf()
    ts_handler.SetTranslation(gp_Vec(vector[0],
                                     vector[1],
                                     vector[2]))
    loc = TopLoc_Location(ts_handler)
    item.Move(loc)

def reverse(item:TopoDS_Shape):
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

def split(item: TopoDS_Shape, 
          nz: int = None, 
          layer_thickness: float = None,
          split_z: bool = True, 
          split_x: bool = False, 
          split_y: bool = False, 
          nx: int = None, 
          ny: int = None):
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
        z_list = np.linspace(0, z, nz)
    if layer_thickness != None:
        z_list = np.arange(0, z, layer_thickness)
    bo = BOPAlgo_Builder()
    bo.AddArgument(item)
    for i in z_list:
        p1, v1 = gp_Pnt(0,0,i), gp_Vec(0, 0, -1)
        fc1 = make_face(gp_Pln(p1, vec_to_dir(v1)), -plan_len, plan_len, -plan_len, plan_len)
        bo.AddArgument(fc1)
    if ny!= None:
        y = ymax - ymin
        y_list = np.linspace(0, y, ny)
        for i in y_list:
            p1, v1 = gp_Pnt(0,i,0), gp_Vec(0, 1, 0)
            fc1 = make_face(gp_Pln(p1, vec_to_dir(v1)), -plan_len, plan_len, -plan_len, plan_len)
            bo.AddArgument(fc1)
    if nx != None:
        x = xmax - xmin
        x_list = np.linspace(0, x, nx)
        for i in x_list:
            p1, v1 = gp_Pnt(i,0,0), gp_Vec(1, 0, 0)
            fc1 = make_face(gp_Pln(p1, vec_to_dir(v1)), -plan_len, plan_len, -plan_len, plan_len)
            bo.AddArgument(fc1)
    bo.Perform()
    top = Topo(bo.Shape())
    geo = geometry_builder(top.solids())
    return geo

def intersector(item: TopoDS_Shape,
                     position: float,
                     axis: str) -> TopoDS_Shape:
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
            p1, v1 = gp_Pnt(0,0,position), gp_Vec(0, 0, 1)
            fc1 = make_face(gp_Pln(p1, vec_to_dir(v1)), -plan_len, plan_len, -plan_len, plan_len)
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
    scaled_shape = BRepBuilderAPI.BRepBuilderAPI_Transform(item, scaling_transform, True).Shape()
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

def rotate_face(shape: TopoDS_Shape, angle: float, axis: str = "z"):
    """
     @brief Rotate the topography by the given angle around the center of mass of the face.
     @param shape TopoDS_Shape to be rotated.
     @param angle Angle ( in degrees ) to rotate by.
     @param axis determine the rotation axis.
     @return the rotated shape.
    """
    transform = gp_Trsf()
    cnt = get_face_center_of_mass(shape, gp_pnt=True)
    match axis:
        case "z":
            ax = gp_Ax1(cnt, gp_Dir(0,0,1))
        case "y":
            ax = gp_Ax1(cnt, gp_Dir(0,1,0))
        case "x":
            ax = gp_Ax1(cnt, gp_Dir(1,0,0))
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

def common(shape1: TopoDS_Shape, shape2: TopoDS_Shape) -> TopoDS_Shape:
    """
     @brief Common between two TopoDS_Shapes. The result is a shape that has all components of shape1 and shape2
     @param shape1 the first shape to be compared
     @param shape2 the second shape to be compared ( must be same shape! )
     @return the common shape or None if there is no common shape between shape1 and shape2 in the sense that both shapes are
    """
    comm = BRepAlgoAPI_Common(shape1, shape2)
    return comm.Shape()