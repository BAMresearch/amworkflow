from OCC.Core.gp import gp_Pnt, gp_Trsf, gp_Vec
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Copy

def translate(item: any,
                vector: list):
    ts_handler = gp_Trsf()
    ts_handler.SetTranslation(gp_Vec(vector[0],
                                     vector[1],
                                     vector[2]))
    loc = TopLoc_Location(ts_handler)
    item.Move(loc)

def reverse(item:any):
    return item.Reversed()
    
def geom_copy(item: any):
    wire_top_builder = BRepBuilderAPI_Copy(item)
    wire_top_builder.Perform(item, True)
    new_item = wire_top_builder.Shape()
    return new_item