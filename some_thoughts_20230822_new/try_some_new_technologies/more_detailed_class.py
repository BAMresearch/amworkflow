from typing import Any
import numpy as np
from amworkflow.src.constants.exceptions import InvalidGeometryException
from amworkflow.src.constants.enums import GEOM_TYPE


'''
0 - pnt
1 - vec
2 - line
3 - segment
4 - point series
5 - polygon
'''
class Service():
    gid_index = {} #geometry object index
    count_go_id = 0 #count geometry object id
    go_property = {}
    go_type_index = GEOM_TYPE
    
    
    # def access_go_type(self):
    #     if isinstance
    @staticmethod
    def value_repetition_check(base_value:any, item_value:any) -> bool:
        '''
        Check if a value is close enough to the base value. True if repeated.
        :param base_value: item to be compared with.
        :param item_value: value to be examined.
        '''
        return np.isclose(np.linalg.norm(base_value-item_value), 0)
    
    @staticmethod
    def type_repetition_check(base_type: int, item_type: int) -> bool:
        valid = item_type in go_type_index
        different = base_type != item_type
        if valid:
            if different:
                return False
            else:
                return True
        else:
            raise Exception("Wrong geometry object type, perhaps a mistake made in development.")
        
    
    def new_item(self, item_value: any, item_type) -> tuple:
        '''
        Check if a value already exits in the index
        :param item_value: value to be examined.
        '''
        for i, v in self.gid_index.items():
            if self.value_repetition_check(v, item_value):
                if self.type_repetition_check(self.go_property[i],item_type):
                    return False, i
        return True, None
    
    def register_item(self, item_value, item_type) -> int:
        '''
        Register an item to the index and return its id. Duplicate value will be filtered.
        :param item_value: value to be registered.
        '''
        new, old_id = self.new_item(item_value, item_type)
        if new:
            self.gid_index.update({self.count_go_id: item_value})
            go_id = self.count_go_id
            self.count_go_id += 1
            return go_id
        else:
            return old_id

S = Service()

class CompObject():
    def __init__(self) -> None:
        pass

class SimpObject():
    def __init__(self) -> None:
        pass
    

class Pnt(SimpObject):
    def __init__(self,coord: list) -> None:
        super().__init__()
        self.go_type: 0
        self.coord = self.pnt(coord)
        Service.register_item(self.coord,"point")
        
    def pnt(self, pt_coord) -> np.ndarray:
        opt = np.array(pt_coord)
        dim = len(pt_coord)
        if dim > 3:
            raise Exception(
                f"Got wrong point {pt_coord}: Dimension more than 3rd provided.")
        if dim < 3:
            opt = np.lib.pad(opt, ((0, 3 - dim)),
                             "constant", constant_values=0)
        return opt
    
    
    
class Vec(SimpObject):
    def __init__(self, pnt1: Pnt, pnt2:Pnt) -> None:
        super().__init__()
        self.go_type: 1
        self.pnt1_coord = pnt1.coord
        self.pnt2_coord = pnt2.coord
        self.value = self.pnt2_coord - self.pnt1_coord
        self.length = np.linalg.norm(self.value)
        self.check_validity()
        self.direction = self.value / self.length
        Service().register_item(self.value)

    def check_validity(self):
        zero_vec = np.isclose(self.length,0)
        if zero_vec:
            raise InvalidGeometryException("vector")
        
class Line(SimpObject):
    def __init__(self,vec:Vec) -> None:
        super().__init__()
        self.go_type: 2
        self.vec = vec
        self.direction = vec.direction
        self.length = np.inf
    
    def pick_point(self,lmbda: float) -> np.ndarray:
        '''
        return a point based on the 1D coordinate lmbda.
        :param lmbda: 1D Coordinate orginates from the start point of the vector.
        '''
        p_coord = self.vec.pnt1_coord + self.direction * lmbda
        return Pnt(p_coord)

class Segment(Vec):
    def __init__(self, pnt1: Pnt, pnt2: Pnt) -> None:
        super().__init__(pnt1, pnt2)
        self.go_type: 3
    
class PntSeries(CompObject):
    def __init__(self,*pnt:Pnt):
        super().__init__()
        self.go_type: 4
        self.pnts = pnt
        self.pts_digraph = {}
        self.coords = self.get_coords()
        self.init_pts_sequence = []
        self.init_pnts()
    
    def get_coords(self):
        if len(self.pnts) == 1 and not isinstance(self.pnts[0],Pnt):
            self.pnts = self.pnts[0]
        coords = [i.coord for i in self.pnts]
        return coords
    
    def init_pnts(self) -> None:
        for i, pt in enumerate(self.coords):
            pt_id = self.register_item(pt)
            if i != len(self.coords) - 1:
                self.init_pts_sequence.append(pt_id)
            if i != 0:
                self.update_digraph(self.init_pts_sequence[i-1], pt_id)
                self.init_pts_sequence[i -
                                       1] = [self.init_pts_sequence[i-1], pt_id]

    def update_digraph(self, start_node: int, end_node: int, insert_node: int = None, build_new_edge: bool = True) -> None:
        if start_node not in self.gid_index:
            raise Exception(f"Unrecognized start node: {start_node}.")
        if end_node not in self.gid_index:
            raise Exception(f"Unrecognized end node: {end_node}.")
        if (insert_node not in self.gid_index) and (insert_node is not None):
            raise Exception(f"Unrecognized inserting node: {insert_node}.")
        if start_node in self.pts_digraph:
            if insert_node is None:
                self.pts_digraph[start_node].append(end_node)
            else:
                end_node_list_index = self.pts_digraph[start_node].index(
                    end_node)
                self.pts_digraph[start_node][end_node_list_index] = insert_node
                if build_new_edge:
                    self.pts_digraph.update({insert_node: [end_node]})
        else:
            if insert_node is None:
                self.pts_digraph.update({start_node: [end_node]})
            else:
                raise Exception("No edge found for insertion option.")
# p = GeomObject()

# p.register_item(2)
# p.register_item(2)
# p.register_item(23)
# print(p.gid_index)
p = Pnt([0,2])
q = Pnt([0,0,4])
# v = Vec(p,q)
# print(v.length)
# l = Line(v)
# print(l.pick_point(22).coord)
pts = PntSeries(p,q)
print(pts.gid_index)
