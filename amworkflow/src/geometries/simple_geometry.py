from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Wire
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Ax2, gp_Dir
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism, BRepPrimAPI_MakeCylinder
from src.geometries.operator import geom_copy, translate, reverse
from src.geometries.builder import geometry_builder

from OCC.Core.GC import GC_MakeArcOfCircle
import math as m


def create_box(length: float, 
               width: float, 
               height: float, 
               radius: float = None,
               alpha: float = None) -> TopoDS_Shape:
    if (radius == None) | (radius == 0):
        return BRepPrimAPI_MakeBox(length, width, height).Shape()
    else:
        if alpha == None:
            alpha = (length / radius)% (m.pi * 2)
        R = radius + (width / 2)
        r = radius - (width / 2)
        print(R - r * m.cos(alpha))
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
        prism = create_prism(wire, [0, 0, height], True).Shape()
        bottom_face = create_face(wire)
        top_face = reverse(create_face(wire_top))
        component = [prism, top_face, bottom_face]
        curve_box = geometry_builder(component)
        return curve_box

def create_cylinder(radius: float, length: float) -> TopoDS_Shape:
    return BRepPrimAPI_MakeCylinder(radius, length).Shape()

def create_prism(wire: TopoDS_Wire,
                 vector: list,
                 copy: bool):
    return BRepPrimAPI_MakePrism(wire, gp_Vec(vector[0],
                                              vector[1],
                                              vector[2]),
                                 copy)

def create_face(wire: TopoDS_Wire):
    return BRepBuilderAPI_MakeFace(wire).Face()
