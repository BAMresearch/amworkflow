from amworkflow.src.geometries.property import get_occ_bounding_box, get_face_center_of_mass
from amworkflow.src.geometries.builder import geometry_builder
from amworkflow.src.geometries.operator import cutter3D
from amworkflow.src.geometries.simple_geometry import create_wire_by_points, create_prism, create_face
from OCC.Core.gp import gp_Pnt
from amworkflow.src.utils.writer import stl_writer
from amworkflow.src.utils.reader import stl_reader
from amworkflow.src.geometries.operator import reverse, scaler
from amworkflow.src.geometries.property import topo_explorer
#import stl file into OCC
imp = stl_reader("/home/yhe/Documents/amworkflow/usecases/param_prism/terrain.stl")
#get the bounding box of the import model
bbox = get_occ_bounding_box(imp)
#explore all edges in the model
edge = topo_explorer(imp, "edge")
xx = []
yy = []
#select all edges on the boundary.
for e in edge:
    xmin, ymin, zmin, xmax, ymax, zmax = get_occ_bounding_box(e)
    if (ymin + ymax < 1e-3) or (abs((ymin + ymax)*0.5 - bbox[4]) < 1e-3):
        xx.append(e)
    if (xmin + xmax < 1e-3) or (abs((xmin + xmax)*0.5 - bbox[3]) < 1e-3):
        yy.append(e)
edges = xx + yy
#build a compound of all edges
wire = geometry_builder(edges)
#get the zmax of the new wire object for creating the basement.
wire_zmax = get_occ_bounding_box(wire)[-1]
prism = reverse(create_prism(wire,[0,0,-wire_zmax], True))
#get the bounding box of the import model
xmin, ymin, zmin, xmax, ymax, zmax = get_occ_bounding_box(imp)
pts = [gp_Pnt(xmin, ymin, 0),
        gp_Pnt(xmax, ymin, 0),
        gp_Pnt(xmax, ymax, 0),
        gp_Pnt(xmin, ymax, 0)]
#create the bottom of the basement
btm_wire = create_wire_by_points(pts)
btm_face = reverse(create_face(btm_wire))
#create a cutter for trimming the prism.
cutter = create_prism(scaler(btm_face, get_face_center_of_mass(btm_face,True),1.2),[0,0,-15],True)
prism = reverse(cutter3D(prism, cutter))
#sew the prism, the bottom face and the imported model together
output = geometry_builder([prism, btm_face, imp])
#output the model.
stl_writer(output, "cutt.stl")
