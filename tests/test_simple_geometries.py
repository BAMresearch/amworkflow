from OCC.Core.TopoDS import TopoDS_Solid
from amworkflow.geometry import simple_geometries


def test_create_box():
    box = simple_geometries.create_box(1.0, 2.0, 3.0)
    assert isinstance(box, TopoDS_Solid)
