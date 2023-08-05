from amworkflow.src.interface.api import amWorkflow as aw
@aw.engine.amworkflow()
def geom_spn(pm):
    imp = pm.imported_model
    bbox = aw.geom.get_occ_bounding_box(imp)
    edge = aw.geom.topo_explorer(imp, "edge")
    xx = []
    yy = []
    #select all edges on the boundary.
    for e in edge:
        xmin, ymin, zmin, xmax, ymax, zmax = aw.geom.get_occ_bounding_box(e)
        if (ymin + ymax < 1e-3) or (abs((ymin + ymax)*0.5 - bbox[4]) < 1e-3):
            xx.append(e)
        if (xmin + xmax < 1e-3) or (abs((xmin + xmax)*0.5 - bbox[3]) < 1e-3):
            yy.append(e)
    edges = xx + yy
    #build a compound of all edges
    wire = aw.geom.geometry_builder(edges)
    #get the zmax of the new wire object for creating the basement.
    wire_zmax = aw.geom.get_occ_bounding_box(wire)[-1]
    prism = aw.geom.reverse(aw.geom.create_prism(wire,[0,0,-wire_zmax], True))
    #get the bounding box of the import model
    xmin, ymin, zmin, xmax, ymax, zmax = aw.geom.get_occ_bounding_box(imp)
    pts = [aw.geom.pnt(xmin, ymin, 0),
            aw.geom.pnt(xmax, ymin, 0),
            aw.geom.pnt(xmax, ymax, 0),
            aw.geom.pnt(xmin, ymax, 0)]
    #create the bottom of the basement
    btm_wire = aw.geom.create_wire_by_points(pts)
    btm_face = aw.geom.reverse(aw.geom.create_face(btm_wire))
    #create a cutter for trimming the prism.
    cutter = aw.geom.create_prism(aw.geom.scale(btm_face, aw.geom.get_face_center_of_mass(btm_face,True),1.2),[0,0,-15],True)
    prism = aw.geom.reverse(aw.geom.cutter3D(prism, cutter))
    #sew the prism, the bottom face and the imported model together
    output = aw.geom.geometry_builder([prism, btm_face, imp])
    return output