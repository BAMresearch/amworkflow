import gmsh
import math as m
from OCC.Core.TopoDS import TopoDS_Shape
from amworkflow.src.geometries.operator import split
from amworkflow.src.geometries.operator import get_occ_bounding_box
from amworkflow.src.constants.exceptions import GmshUseBeforeInitializedException
import logging

def gmsh_switch(s: bool) -> None:
    if s:
        gmsh.initialize()
    else:
        gmsh.finalize()

def get_geom_pointer(model: gmsh.model, shape: TopoDS_Shape) -> list:
    try:
        gmsh.is_initialized()
    except:
        raise GmshUseBeforeInitializedException()
    return model.occ.importShapesNativePointer(int(shape.this), highestDimOnly=True)
    
def mesher(item: TopoDS_Shape,
           model_name: str,
           layer_type: bool,
           layer_param : float = None,
           size_factor: float = 0.1):
    try:
        gmsh.is_initialized()
    except:
        raise GmshUseBeforeInitializedException()
    if layer_type: 
        geo = split(item=item,
            split_z=True,
            layer_thickness= layer_param)
    else:
        geo = split(item=item,
            split_z=True,
            nz = layer_param)
    model = gmsh.model()
    model.add(model_name)
    v = get_geom_pointer(model, geo)
    model.occ.synchronize()
    for layer in v:
        model.add_physical_group(3,[layer[1]], name=f"layer{layer[1]}")
        phy_gp = model.getPhysicalGroups()
    gmsh.option.setNumber("Mesh.MeshSizeFactor", size_factor)
    model.mesh.generate()
    model.mesh.remove_duplicate_nodes()
    model.mesh.remove_duplicate_elements()
    phy_gp = model.getPhysicalGroups()
    model_name = model.get_current()
    return model

