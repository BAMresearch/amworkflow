# add function you actually used from old simple_geometry.py
import math as m

from OCC.Core.BRepBuilderAPI import (BRepBuilderAPI_MakeEdge,
                                     BRepBuilderAPI_MakeFace,
                                     BRepBuilderAPI_MakeSolid,
                                     BRepBuilderAPI_MakeWire,
                                     BRepBuilderAPI_Sewing)
from OCC.Core.BRepPrimAPI import (BRepPrimAPI_MakeBox,
                                  BRepPrimAPI_MakePrism)
from OCC.Core.GC import GC_MakeArcOfCircle
from OCC.Core.gp import gp_Pnt, gp_Vec
from OCC.Core.TopoDS import (TopoDS_Face, TopoDS_Shape, TopoDS_Shell,
                             TopoDS_Solid, TopoDS_Wire)
from OCCUtils.Topology import Topo

from amworkflow.geometry import helpers

#(geom_copy, geometry_builder, reverse,
#                                        sewer, translate)


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
            sewed_face = helper.sewer(faces)
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
        wire_top = helpers.geom_copy(wire)
        helpers.translate(wire_top, [0, 0, height])
        prism = create_prism(wire, [0, 0, height], True)
        bottom_face = create_face(wire)
        top_face = helpers.reverse(create_face(wire_top))
        component = [prism, top_face, bottom_face]
        sewing = BRepBuilderAPI_Sewing()
        # Add all components to the sewing.
        for i in range(len(component)):
            sewing.Add(component[i])
        sewing.Perform()
        sewed_shape = sewing.SewedShape()
        # shell = BRepBuilderAPI_MakeShell(sewed_shape)
        solid = BRepBuilderAPI_MakeSolid(sewed_shape).Shape()
        curve_box = helpers.geometry_builder(component)
        # Returns the shape of the shell.
        if shell:
            return sewed_shape
        else:
            return solid

def create_face(wire: TopoDS_Wire) -> TopoDS_Face:
    """
     @brief Create a BRep face from a TopoDS_Wire. This is a convenience function to use : func : ` BRepBuilderAPI_MakeFace ` and
     @param wire The wire to create a face from. Must be a TopoDS_Wire.
     @return A Face object with the properties specified
    """
    return BRepBuilderAPI_MakeFace(wire).Face()

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

