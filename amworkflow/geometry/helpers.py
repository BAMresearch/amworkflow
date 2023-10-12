# helper functions needed for geometry building

from OCC.Core.BRep import BRep_Builder
from OCC.Core.BRepBuilderAPI import (BRepBuilderAPI_Copy,
                                     BRepBuilderAPI_MakeSolid,
                                     BRepBuilderAPI_Sewing)
from OCC.Core.gp import gp_Trsf, gp_Vec
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Shape


def geometry_builder(*args):
    builder = BRep_Builder()
    obj = TopoDS_Compound()
    builder.MakeCompound(obj)
    for item in args[0]: builder.Add(obj, item)
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