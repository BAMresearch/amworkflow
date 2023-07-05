from OCC.Core.gp import gp_Pnt, gp_Trsf, gp_Vec
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Copy
from OCC.Core.TopoDS import TopoDS_Shape
import numpy as np
from OCC.Core.gp import gp_Pln
from OCC.Core.gp import gp_Pnt, gp_Vec
from OCC.Core.BOPAlgo import BOPAlgo_Builder
from OCCUtils.Topology  import Topo
from OCCUtils.Construct import make_face
from OCCUtils.Construct import vec_to_dir
from amworkflow.src.geometries.builder import geometry_builder
from OCC.Core.BRepBndLib import brepbndlib_Add
from OCC.Core.Bnd import Bnd_Box

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

def get_occ_bounding_box(shape: TopoDS_Shape):
    bbox = Bnd_Box()
    add_bbox = brepbndlib_Add(shape, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    return xmin, ymin, zmin, xmax, ymax, zmax

def split(item: TopoDS_Shape, 
          nz: int = None, 
          layer_thickness: float = None,
          split_z: bool = True, 
          split_x: bool = False, 
          split_y: bool = False, 
          nx: int = None, 
          ny: int = None):
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
    