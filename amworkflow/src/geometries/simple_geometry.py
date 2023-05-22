from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Core.TColgp import TColgp_Array1OfPnt
from OCC.Core.gp import gp_Pnt, gp_Vec
from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism, BRepPrimAPI_MakeCylinder
from OCC.Core.BRepFilletAPI import BRepFilletAPI_MakeFillet


def create_box(length, width, height) -> TopoDS_Shape:
    return BRepPrimAPI_MakeBox(length, height, width).Shape()

def create_cylinder(radius: float, length: float) -> TopoDS_Shape:
    return BRepPrimAPI_MakeCylinder(radius, length).Shape()

def create_prism():
    array = TColgp_Array1OfPnt(1, 5)
    array.SetValue(1, gp_Pnt(0, 0, 0))
    array.SetValue(2, gp_Pnt(0, 2, 0))
    array.SetValue(3, gp_Pnt(2, 2, 0))
    array.SetValue(4, gp_Pnt(4, 0, 0))
    array.SetValue(5, gp_Pnt(0, 0, 0))
    bspline = GeomAPI_PointsToBSpline(array).Curve()
    profile = BRepBuilderAPI_MakeEdge(bspline).Edge()
    starting_point = gp_Pnt(0.0, 0.0, 0.0)
    end_point = gp_Pnt(0.0, 0.0, 6.0)
    vec = gp_Vec(starting_point, end_point)
    path = BRepBuilderAPI_MakeEdge(starting_point, end_point).Edge()
    return BRepPrimAPI_MakePrism(profile, vec).Shape()