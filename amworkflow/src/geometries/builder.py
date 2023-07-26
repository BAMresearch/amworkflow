from OCC.Core.BRep import BRep_Builder
from OCC.Core.TopoDS import TopoDS_Compound
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Wire, TopoDS_Shell
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