# add function you actually used from old simple_geometry.py
import math as m

from OCC.Core.BRepBuilderAPI import (BRepBuilderAPI_MakeEdge, 
                                     BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace, BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid, BRepBuilderAPI_MakeShell, brepbuilderapi_Precision, BRepBuilderAPI_MakePolygon, BRepBuilderAPI_Copy)
from OCC.Core.BRepPrimAPI import (BRepPrimAPI_MakeBox,
                                  BRepPrimAPI_MakePrism)
from OCC.Core.GC import GC_MakeArcOfCircle
from OCC.Core.Geom import Geom_TrimmedCurve
from OCC.Core.BRep import BRep_Builder
from OCC.Core.gp import (gp_Pln, 
                         gp_Pnt, 
                         gp_Trsf, 
                         gp_Vec)
from OCC.Core.TopoDS import (TopoDS_Face, 
                             TopoDS_Shape,
                             TopoDS_Edge, 
                             TopoDS_Shell,
                             TopoDS_Solid,
                             TopoDS_Wire,
                             TopoDS_Compound)
from OCCUtils.Topology import Topo

from amworkflow import occ_helpers
import numpy as np
from amworkflow.geometry import builtinCAD as bcad


#(geom_copy, geometry_builder, reverse,
#                                        sewer, translate)
def create_edge(pnt1: gp_Pnt = None, pnt2: gp_Pnt = None, arch: Geom_TrimmedCurve = None) -> TopoDS_Edge:
    """
     @brief Create an edge between two points. This is a convenience function to be used in conjunction with : func : ` BRepBuilderAPI_MakeEdge `
     @param pnt1 first point of the edge
     @param pnt2 second point of the edge
     @param arch arch edge ( can be None ). If arch is None it will be created from pnt1 and pnt2
     @return an edge.
    """
    if isinstance(pnt1, gp_Pnt) and isinstance(pnt2, gp_Pnt):
        edge = BRepBuilderAPI_MakeEdge(pnt1, pnt2).Edge()
    elif isinstance(arch, Geom_TrimmedCurve):
        edge = BRepBuilderAPI_MakeEdge(arch).Edge()
    return edge

def create_wire(*edge) -> TopoDS_Wire:
    """
     @brief Create a wire. Input at least one edge to build a wire. This is a convenience function to call BRepBuilderAPI_MakeWire with the given edge and return a wire.
     @return A wire built from the given edge ( s ). The wire may be used in two ways : 1
    """
    return BRepBuilderAPI_MakeWire(*edge).Wire()

def create_face(wire: TopoDS_Wire) -> TopoDS_Face:
    """
     @brief Create a BRep face from a TopoDS_Wire. This is a convenience function to use : func : ` BRepBuilderAPI_MakeFace ` and
     @param wire The wire to create a face from. Must be a TopoDS_Wire.
     @return A Face object with the properties specified
    """
    return BRepBuilderAPI_MakeFace(wire).Face()

def create_solid(item: TopoDS_Shape) -> TopoDS_Shape:
    return BRepBuilderAPI_MakeSolid(item).Shape()

def create_compound(*args):
    builder = BRep_Builder()
    obj = TopoDS_Compound()
    builder.MakeCompound(obj)
    for item in args[0]: builder.Add(obj, item)
    return obj

def create_solid(item: TopoDS_Shape) -> TopoDS_Shape:
    return BRepBuilderAPI_MakeSolid(item).Shape()



def create_box(length: float,
               width: float,
               height: float,
               radius: float = None,
               alpha: float = None,
               shell: bool = False) -> TopoDS_Shape:
    """
    @brief Create a box with given length width height and radius. If radius is None or 0 the box will be sewed by a solid.
    @param length Length of the box in points
    @param width Width of the box.
    @param height Height of the box.
    @param radius Radius of the box. Default is None which means that the box is without curves.
    @param alpha defines the angle of bending the box. Default is half the length divided by the radius.
    @param shell If True the box will be shell. Default is False.
    @return TopoDS_Shape with box in it's topolar form. Note that this is a Shape
    """
    if (radius == None) or (radius == 0):
        if shell:
            box = BRepPrimAPI_MakeBox(length, width, height).Shape()
            faces = list(Topo(TopoDS_Solid).faces_from_solids(box))
            print(isinstance(faces[0], TopoDS_Shape))
            sewed_face = occ_helpers.sew_face(faces)
            # make_shell = BRepBuilderAPI_MakeShell(sewed_face, False).Shell()

            return sewed_face
        else:
            return BRepPrimAPI_MakeBox(length, width, height).Shape()
    else:
        # The alpha of the circle.
        if alpha == None:
            alpha = (length / radius) % (m.pi * 2)
        R = radius + (width / 2)
        r = radius - (width / 2)
        p1 = gp_Pnt(0, 0, 0)
        p1_2 = gp_Pnt((1 - m.cos(0.5 * alpha)) * R, R * m.sin(0.5 * alpha), 0)
        p2 = gp_Pnt((1 - m.cos(alpha)) * R, R * m.sin(alpha), 0)
        p3 = gp_Pnt(R - r * m.cos(alpha), r * m.sin(alpha), 0)
        p3_4 = gp_Pnt(R - r * m.cos(0.1 * alpha), r * m.sin(0.1 * alpha), 0)
        p4 = gp_Pnt(width, 0, 0)
        arch1_2 = GC_MakeArcOfCircle(p1, p1_2, p2)
        arch3_4 = GC_MakeArcOfCircle(p3, p3_4, p4)
        arch_edge1_2 = BRepBuilderAPI_MakeEdge(arch1_2.Value()).Edge()
        arch_edge3_4 = BRepBuilderAPI_MakeEdge(arch3_4.Value()).Edge()
        edge2 = BRepBuilderAPI_MakeEdge(p2, p3).Edge()
        edge4 = BRepBuilderAPI_MakeEdge(p4, p1).Edge()
        wire = BRepBuilderAPI_MakeWire(
            arch_edge1_2, edge2, arch_edge3_4, edge4).Wire()
        wire_top = occ_helpers.geom_copy(wire)
        occ_helpers.translate(wire_top, [0, 0, height])
        prism = create_prism(wire, [0, 0, height], True)
        bottom_face = create_face(wire)
        top_face = occ_helpers.reverse(create_face(wire_top))
        component = [prism, top_face, bottom_face]
        sewing = BRepBuilderAPI_Sewing()
        # Add all components to the sewing.
        for i in range(len(component)):
            sewing.Add(component[i])
        sewing.Perform()
        sewed_shape = sewing.SewedShape()
        # shell = BRepBuilderAPI_MakeShell(sewed_shape)
        solid = BRepBuilderAPI_MakeSolid(sewed_shape).Shape()
        print(solid)
        curve_box = create_compound(component)
        
        # Returns the shape of the shell.
        if shell:
            return sewed_shape
        else:
            return solid



def create_prism(shape: TopoDS_Shape,
                 vector: list,
                 copy: bool = True) -> TopoDS_Shell:
    """
    @brief Create prism from TopoDS_Shape and vector. It is possible to copy the based wire(s) if copy is True. I don't know what if it's False so it is recommended to always use True.
    @param shape TopoDS_Shape to be used as base
    @param vector list of 3 elements ( x y z ). Normally only use z to define the height of the prism.
    @param copy boolean to indicate if the shape should be copied
    @return return the prism
    """
    return BRepPrimAPI_MakePrism(shape, gp_Vec(vector[0],
                                               vector[1],
                                               vector[2]),
                                 copy).Shape()

class Pnt():
    def __init__(self, *coords: list):
        self.coords = coords
        if (len(self.coords) == 1) and (type(self.coords[0]) is list or isinstance(self.coords[0],np.ndarray)):
            self.coords = coords[0]
        self.coords = self.format_coords()
        self.pts_index = {}
        self.pts_digraph = {}
        self.count_pt_id = 0
        self.init_pts_sequence = []
        self.init_pnts()

    def enclose(self):
        distance = np.linalg.norm(self.coords_numpy[-1] - self.coords_numpy[0])
        if np.isclose(distance, 0):
            print("Polygon seems already enclosed, skipping...")
        else:
            self.coords_numpy = np.vstack(
                (self.coords_numpy, self.coords_numpy[0]))
            self.create_attr()

    # def create(self):
    #     self.coords_numpy = self.create_pnts()
    #     self.eliminate_overlap()
    #     self.create_attr()
    def format_coords(self):
        return [self.pnt(i) for i in self.coords]

    def create_attr(self):
        self.coords_to_list = self.coords_numpy.tolist()
        self.coords_to_gp_Pnt: list
        self.x, self.y, self.z = self.coords_numpy.T
        self.pts_num = self.coords_numpy.shape[0]

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

    def new_pnt(self, pt_coords: list):
        pt_coords = self.pnt(pt_coords)
        for i, v in self.pts_index.items():
            if self.pnt_overlap(v, pt_coords):
                return False, i
        return True, None

    def pnt_overlap(self, pt1: np.ndarray, pt2: np.ndarray) -> bool:
        return np.isclose(np.linalg.norm(pt1-pt2), 0)

    def init_pnts(self) -> None:
        for i, pt in enumerate(self.coords):
            pt_id = self.register_pnt(pt)
            if i != len(self.coords) - 1:
                self.init_pts_sequence.append(pt_id)
            if i != 0:
                self.update_digraph(self.init_pts_sequence[i-1], pt_id)
                self.init_pts_sequence[i -
                                       1] = [self.init_pts_sequence[i-1], pt_id]

    def register_pnt(self, pt: list) -> int:
        pnt = self.pnt(pt)
        new, old_id = self.new_pnt(pnt)
        if new:
            self.pts_index.update({self.count_pt_id: pnt})
            pnt_id = self.count_pt_id
            self.count_pt_id += 1
            return pnt_id
        else:
            return old_id

    def update_digraph(self, start_node: int, end_node: int, insert_node: int = None, build_new_edge: bool = True) -> None:
        if start_node not in self.pts_index:
            raise Exception(f"Unrecognized start node: {start_node}.")
        if end_node not in self.pts_index:
            raise Exception(f"Unrecognized end node: {end_node}.")
        if (insert_node not in self.pts_index) and (insert_node is not None):
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
            
    
class Segments(Pnt):
    def __init__(self, *coords: list):
        super().__init__(*coords)
        self.segments_index = {}
        self.modify_edge_list = {}
        self.virtual_vector = {}
        self.virtual_pnt = {}
        self.find_overlap_node_on_edge()
        self.modify_edge()

    def enclose(self):
        pass

    def get_segment(self, pt1: int, pt2: int) -> np.ndarray:
        return np.array([self.pts_index[pt1], self.pts_index[pt2]])

    # def init_segments(self):
        for i, v in enumerate(self.pts_sequance):
            if i != len(self.pts_sequance) - 1:
                self.segments_index.update({v: [self.pts_sequance[i+1]]})
                self.count_vector_id += 1

    def insert_item(self, *items: np.ndarray, original: np.ndarray, insert_after: int) -> np.ndarray:
        print(original[:insert_after+1])
        return np.concatenate((original[:insert_after+1], items, original[insert_after+1:]))
    
    def add_pending_change(self, edge: tuple, new_node: int) -> None:
        if edge in self.modify_edge_list:
            self.modify_edge_list[edge].append(new_node)
        else:
            self.modify_edge_list.update({edge:[new_node]})
            
    def modify_edge(self):
        for edge,nodes in self.modify_edge_list.items():
            edge_0_coords = self.pts_index[edge[0]]
            nodes_coords = [self.pts_index[i] for i in nodes]
            distances = [np.linalg.norm(i-edge_0_coords) for i in nodes_coords]
            order = np.argsort(distances)
            nodes = [nodes[i] for i in order]
            self.pts_digraph[edge[0]].remove(edge[1])
            pts_list = [edge[0]]+nodes+[edge[1]]
            for i,nd in enumerate(pts_list):
                if i == 0:
                    continue
                self.update_digraph(pts_list[i-1],nd,build_new_edge=False)
                if (i != 1) and (i != len(pts_list) - 1):
                    if (pts_list[i-1] in self.virtual_pnt) and nd in (self.virtual_pnt):
                        self.virtual_vector.update({(pts_list[i-1],nd): True})
    
    def check_self_edge(self, line: np.ndarray) -> bool:
        if self.pnt_overlap(line[0],line[1]):
            return True
        else:
            return False
                    
    def overlap_node_on_edge_finder(self,i,j):
        v = self.init_pts_sequence[i]
        vv = self.init_pts_sequence[j]
        print(i,j)
        lin1 = self.get_segment(v[0], v[1])
        lin2 = self.get_segment(vv[0], vv[1])
        self_edge = (self.check_self_edge(lin1) or self.check_self_edge(lin2))
        if not self_edge:
            parallel, colinear = bcad.check_parallel_line_line(lin1, lin2)
            # if v == [13,14]:
            #     print("line:",(v,vv), parallel, colinear)
            if parallel:
                if colinear:
                    index, coords = bcad.check_overlap(lin1, lin2)
                    if len(index) < 4:
                        for ind in index:
                            if ind in [0,1]:
                                self.add_pending_change(tuple(vv),v[ind])
                            else:
                                self.add_pending_change(tuple(v),vv[ind])
            else:
                distance = bcad.shortest_distance_line_line(lin1, lin2)
                intersect = np.isclose(distance[0],0)
                new,pt = self.new_pnt(distance[1][0])
                if intersect and new:
                    pnt_id = self.register_pnt(distance[1][0])
                    self.add_pending_change(tuple(v),pnt_id)
                    self.add_pending_change(tuple(vv), pnt_id)
        

    def arg_generator(self):
        iter_range = range(len(self.init_pts_sequence))
        visited = {}
        for i in iter_range:
            for j in iter_range:
                if i == j:
                    continue
                if i == j + 1:
                    continue
                if j == i - 1:
                    continue
                if (i,j) in visited or (j,i) in visited:
                    continue
                args = (self,i,j)
                print(args)
                yield args
            visited.update({(i,j): True, (j,i): True})
        
    
    def find_overlap_node_on_edge(self):
        visited = {}
        for i,v in enumerate(self.init_pts_sequence):
            for j,vv in enumerate(self.init_pts_sequence):
                if i == j:
                    continue
                if i == j + 1:
                    continue
                if j == i + 1:
                    continue
                if i == len(self.init_pts_sequence) * 2 -1 - i:
                    continue
                if (i,j) in visited or (j,i) in visited:
                    continue
                print(i,j)
                lin1 = self.get_segment(v[0], v[1])
                lin2 = self.get_segment(vv[0], vv[1])
                self_edge = (self.check_self_edge(lin1) or self.check_self_edge(lin2))
                if not self_edge:
                    parallel, colinear = bcad.check_parallel_line_line(lin1, lin2)
                    # if v == [13,14]:
                    #     print("line:",(v,vv), parallel, colinear)
                    if parallel:
                        if colinear:
                            index, coords = bcad.check_overlap(lin1, lin2)
                            if len(index) < 4:
                                for ind in index:
                                    if ind in [0,1]:
                                        self.add_pending_change(tuple(vv),v[ind])
                                    else:
                                        self.add_pending_change(tuple(v),vv[ind-2])
                    else:
                        distance = bcad.shortest_distance_line_line(lin1, lin2)
                        intersect = np.isclose(distance[0],0)
                        new,pt = self.new_pnt(distance[1][0])
                        if intersect and new:
                            pnt_id = self.register_pnt(distance[1][0])
                            self.virtual_pnt.update({pnt_id: True})
                            self.add_pending_change(tuple(v),pnt_id)
                            self.add_pending_change(tuple(vv), pnt_id)
                visited.update({(i,j): True, (j,i): True})