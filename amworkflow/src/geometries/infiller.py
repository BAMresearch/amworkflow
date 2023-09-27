import numpy as np
from OCC.Core.TopoDS import TopoDS_Shape
from amworkflow.src.constants.exceptions import DimensionViolationException
from amworkflow.src.geometries.composite_geometry import polygon_maker, isoceles_triangle_maker
class PolygonInfiller(object):
    def __init__(self) -> None:
        self.pbox_ln: float
        self.pbox_wd: float
        self.side_len: float
        self.bbox_ratio: float
    
    def config(self, bbox_len: float, bbox_wid: float, bbox_hgt: float, side_num: float, infill_rate: float = None):
        self.bbox_len = bbox_len
        self.bbox_wid = bbox_wid
        self.bbox_hgt = bbox_hgt
        self.side_num = side_num
        if (infill_rate < 1) and (infill_rate > 0):
            self.infill_rate = infill_rate  
        else:
            raise DimensionViolationException("Invalid infill rate.")
        self.bbox_ratio = self.bbox_wid / self.bbox_hgt
         
    def rglr_plygn_side_len_handler(self):
        match self.side_num:
            case 3:
                ln = self.pbox_ln
                wd = 0.5*(3**0.5)*self.ln
            case 4:
                ln = self.pbox_ln
                wd = ln
            case 5:
                ln = self.pbox_ln / (2 * np.sin(np.deg2rad(18)) + 1)
                wd = ln * (np.sin(np.deg2rad(18)) + np.cos(np.deg2rad(18)))
            case 6:
                ln = self.pbox_ln / 2
                wd = ln
        if abs(ln / wd -self.pbox_ln / self.pbox_wd) > 1e-3:
            return self.pbox_ln, self.pbox_wd, False
        else:
            return ln, wd, True
    
    def raid_creator(self):
        pass      
                

    # def polygon_infiller(side_num: int, side_len:float, bbox_len:float, bbox_wid:float,bbox_hgt:float) -> TopoDS_Shape: