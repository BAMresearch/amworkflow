from amworkflow.src.geometries.operator import intersector
from amworkflow.src.geometries.simple_geometry import create_box
from amworkflow.src.utils.writer import stl_writer
from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Iterator
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_ShapeEnum
from OCC.Core.TopoDS import TopoDS_Face
from OCC.Core.BRepTools import BRepTools_WireExplorer
from OCC.Core.BRepGProp import brepgprop_VolumeProperties, brepgprop_SurfaceProperties
import OCC.Core.gp as gp
from OCC.Core.GProp import GProp_GProps
com = [intersector(create_box(1,2,3, 80), 1.3, "z")]
# stl_writer(com, "intersct")



# Iterate through the subshapes in the compound
def subshape(shape_list):
    for shape in shape_list:
        iterator = TopoDS_Iterator(shape)
        while iterator.More():
            subshape = iterator.Value()
            
            # Do something with the subshape
            # For example, you can print its shape type
            shape_type = subshape.ShapeType()
            print(f"Shape type: {shape_type}")
            opt_slist = []
            opt_slist.append(subshape)
            
            # Move to the next subshape
            iterator.Next()
    return opt_slist



