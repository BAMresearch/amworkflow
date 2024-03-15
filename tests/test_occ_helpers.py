from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Shape, TopoDS_Solid

from amworkflow import occ_helpers as occh


def test_solid_maker():
    box = occh.create_box(1.0, 2.0, 3.0)
    solid = occh.create_solid(box)
    assert isinstance(solid, TopoDS_Solid)


def test_split_bynumber():
    box = occh.create_box(1.0, 2.0, 3.0)
    geo = occh.split_by_plane(item=box, nz=2, nx=2, ny=2)
    assert isinstance(geo, TopoDS_Compound)


def test_split_bylayerheight():
    box = occh.create_box(1.0, 2.0, 3.0)
    geo = occh.split_by_plane(item=box, layer_height=0.5)
    assert isinstance(geo, TopoDS_Compound)


###
# if __name__ == "__main__":
#     test_split_bylayerheight()
