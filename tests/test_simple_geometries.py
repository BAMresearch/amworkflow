from OCC.Core.TopoDS import TopoDS_Solid

from amworkflow import occ_helpers as occh


def test_create_box():
    box = occh.create_box(1.0, 2.0, 3.0)
    assert isinstance(box, TopoDS_Solid)
