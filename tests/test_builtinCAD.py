import numpy as np
from OCC.Core.gp import gp_Pnt

from amworkflow.geometry.builtinCAD import DuplicationCheck, Pnt, Segment, pnt


def test_duplication_check():
    pnt1 = Pnt([2, 3])
    pnt2 = Pnt([2, 3, 3])
    pnt3 = Pnt([2, 3, 5])
    pnt31 = Pnt([2, 3, 5])
    seg1 = Segment(pnt1, pnt2)
    seg11 = Segment(pnt1, pnt2)
    assert seg1 is seg11
    assert pnt3.id == pnt31.id
    assert pnt1 is not pnt2


def test_pnt_coord_auto_completion():
    point1 = pnt([2, 3])
    point2 = pnt([])
    point3 = pnt([2])
    assert np.array_equal(point1, np.array([2, 3, 0]))
    assert np.array_equal(point2, np.array([0, 0, 0]))
    assert np.array_equal(point3, np.array([2, 0, 0]))


def test_pnt():
    pnt1 = Pnt([2, 3])
    assert np.array_equal(pnt1.value, np.array([2, 3, 0]))
    assert pnt1.type == 0
    assert isinstance(pnt1.occ_pnt, gp_Pnt)
