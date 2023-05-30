from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Core.TColgp import TColgp_Array1OfPnt
from OCC.Core.gp import gp_Pnt, gp_Vec
from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism, BRepPrimAPI_MakeCylinder
from OCC.Core.BRepFilletAPI import BRepFilletAPI_MakeFillet
from OCC.Core.GC import GC_MakeArcOfCircle
# from OCC.Display.backend import load_backend
# load_backend('qt-pyqt5')  # Specify the backend here (e.g., 'qt-pyqt5', 'qt-pyside2', etc.)
from OCC.Display.SimpleGui import init_display
import math


def create_box(length, width, height, radius) -> TopoDS_Shape:
    return BRepPrimAPI_MakeBox(length, width, height).Shape()

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

def create_sweep():
    radius = 5.0  # Radius of the arc

    # Define the profile curve
    profile_center = gp_Pnt(0, 0, 0)
    profile_radius = radius
    profile_start_angle = 0  # Starting angle of the arc
    profile_end_angle = math.pi/2  # Ending angle of the arc

    profile_arc = GC_MakeArcOfCircle(profile_center, profile_radius, profile_start_angle, profile_end_angle)

    profile_wire_builder = BRepBuilderAPI_MakeWire()
    profile_wire_builder.Add(profile_arc.Value())

    profile_wire = profile_wire_builder.Wire()

    # Define the path curve
    path_radius = radius  # Radius of the path
    path_center = gp_Pnt(0, 0, 0)

    path_arc = GC_MakeArcOfCircle(path_center, path_radius, 0, 2 * math.pi)

    path_wire_builder = BRepBuilderAPI_MakeWire()
    path_wire_builder.Add(path_arc.Value())

    path_wire = path_wire_builder.Wire()

    # Create the sweep
    sweep_builder = BRepBuilderAPI_MakeFace(profile_wire, True)
    sweep_shape = sweep_builder.Shape()

    swept_shape_builder = BRepPrimAPI_MakePrism(sweep_shape, path_wire)
    sweep_shape = swept_shape_builder.Shape()
    display, start_display, add_menu, add_function_to_menu = init_display()
    display.DisplayShape(sweep_shape)
    start_display()