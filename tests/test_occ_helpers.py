from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Solid, TopoDS_Compound

from amworkflow.geometry import simple_geometries
from amworkflow import occ_helpers


def test_solid_maker():
    box = simple_geometries.create_box(1., 2., 3.)
    solid = occ_helpers.solid_maker(box)
    assert isinstance(solid, TopoDS_Solid)

def test_split_bynumber():
    box = simple_geometries.create_box(1., 2., 3.)
    geo = occ_helpers.split(item=box, nz = 2, nx= 2, ny = 2)
    assert isinstance(geo, TopoDS_Compound)

def test_split_bylayerheight():
    box = simple_geometries.create_box(1., 2., 3.)
    geo = occ_helpers.split(item=box, layer_height=0.5)
    assert isinstance(geo, TopoDS_Compound)


###
# if __name__ == "__main__":
#     test_split_bylayerheight()