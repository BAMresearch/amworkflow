import amworkflow.geometry.builtinCAD as bcad
from pprint import pprint
class CreateWallByPoints():
    def __init__(self, *pnts, thickness: float, height: float = 0, radius: float = None) -> None:
        self.segments = SegmentHandler(*pnts)
        self.R = radius
        self.th = thickness
        self.h = height

class PntHandler():
    def __init__(self) -> None:
        self.pnts = []
        self.pnt_ids = []
        self.pnt_coords = []
        self.pnt_property = {}
        
    def init_pnts(self, *pnts) -> None:
        self.pnt_ids.extend([item.id for item in pnts])
        self.pnt_coords.extend([item.value for item in pnts])
        # self.pnt_property.update({key:item for key, item in bcad.id_index.items() if item["type"] == 0 and item["id"] in self.pnt_ids})
        
    def init_center_points(self, *pnts) -> None:
        self.init_pnts(*pnts)
        for pnt in pnts:
            pnt.enrich_property({"CWBP": {"center_point": True}})
    
    def handle_boundary_point(self, center_point: bcad.Pnt, boundary_point: bcad.Pnt) -> None:
        center_point.property["CWBP"].update({"derive": boundary_point.id})
        boundary_point.enrich_property({"CWBP": {"center_point": False}})
        boundary_point.property["CWBP"].update({"originate": center_point.id})
            
class SegmentHandler(PntHandler):
    def __init__(self, *pnts: bcad.Pnt) -> None:
        super().__init__()
        self.segments_init = []
        self.digraph = {}
        
    def init_boundary(self) -> None:
        for i, pt in enumerate(self.pnt_ids):
            if i != len(self.pnt_ids) - 1:
                self.segments_init.append(pt)
            if i != 0:
                self.update_digraph(self.segments_init[i-1], pt)
                self.segments_init[i -
                                    1] = [self.segments_init[i-1], pt]
                    
    def update_digraph(self, start_node: int, end_node: int, insert_node: int = None, build_new_edge: bool = True) -> None:
        '''
        Update the digraph of the points.
        '''
        if start_node not in self.pnt_ids:
            raise Exception(f"Unrecognized start node: {start_node}.")
        if end_node not in self.pnt_ids:
            raise Exception(f"Unrecognized end node: {end_node}.")
        if (insert_node not in self.pnt_ids) and (insert_node is not None):
            raise Exception(f"Unrecognized inserting node: {insert_node}.")
        if start_node in self.digraph:
            if insert_node is None:
                self.digraph[start_node].append(end_node)
            else:
                end_node_list_index = self.digraph[start_node].index(
                    end_node)
                self.digraph[start_node][end_node_list_index] = insert_node
                if build_new_edge:
                    self.digraph.update({insert_node: [end_node]})
        else:
            if insert_node is None:
                self.digraph.update({start_node: [end_node]})
            else:
                raise Exception("No edge found for insertion option.")
            
    def calculate_centerline(self, R) -> None:
        '''
        Calculate the centerline of the wall.
        '''
        if R is not None:
            # TODO: implement the interpolation of the centerline.
            pass
        self.hth = 0.5
        # for i,p in enumerate(self.pnt_coords):
    
    def calculate_boundary(self) -> None:
        '''
        Calculate the boundary of the wall.
        '''

pnt1 = bcad.Pnt([2,3])
pnt2 = bcad.Pnt([2,3,3])
pnt3 = bcad.Pnt([2,3,5])
pnt4 = bcad.Pnt([2,3,8])
seg1 = bcad.Segment(pnt1, pnt2)
seg2 = bcad.Segment(pnt2, pnt3)
seg3 = bcad.Segment(pnt3, pnt1)
wire1 = bcad.Wire(seg1, seg2,seg3)
surf1 = bcad.Surface(wire1)
# pprint(bcad.id_index)
# print(seg3)
# print(pnt1.property["occ_pnt"])
pnts_handler = PntHandler()
pnts_handler.init_center_points(pnt1, pnt2, pnt3)
pnts_handler.handle_boundary_point(pnt1, pnt4)
pprint(bcad.id_index)