import multiprocessing
from pprint import pprint

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import OCC.Core.BRepBuilderAPI as BRepBuilderAPI
import OCC.Core.gp as gp
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BOPAlgo import BOPAlgo_Builder, BOPAlgo_MakerVolume, BOPAlgo_Splitter
from OCC.Core.BRep import BRep_Builder
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCC.Core.BRepBndLib import brepbndlib_Add
from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_Copy,
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakePolygon,
    BRepBuilderAPI_MakeShell,
    BRepBuilderAPI_MakeSolid,
    BRepBuilderAPI_MakeWire,
    BRepBuilderAPI_Sewing,
    BRepBuilderAPI_Transform,
    brepbuilderapi_Precision,
)
from OCC.Core.BRepClass3d import BRepClass3d_Intersector3d
from OCC.Core.BRepGProp import brepgprop_SurfaceProperties, brepgprop_VolumeProperties
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakePrism
from OCC.Core.gp import gp_Ax1, gp_Dir, gp_Pln, gp_Pnt, gp_Trsf, gp_Vec
from OCC.Core.GProp import GProp_GProps
from OCC.Core.TopAbs import (
    TopAbs_COMPOUND,
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_FORWARD,
    TopAbs_SHELL,
    TopAbs_SOLID,
    TopAbs_WIRE,
)
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopoDS import (
    TopoDS_Compound,
    TopoDS_Face,
    TopoDS_Iterator,
    TopoDS_Shape,
    TopoDS_Shell,
    TopoDS_Solid,
    TopoDS_Wire,
    topods_Face,
)
from OCC.Core.TopTools import TopTools_ListOfShape
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCCUtils.Construct import make_face, vec_to_dir
from OCCUtils.Topology import Topo

import amworkflow.geometry.builtinCAD as bcad
from amworkflow import occ_helpers as occh


class Pnt:
    def __init__(self, *coords: list):
        self.coords = coords
        if (len(self.coords) == 1) and (
            type(self.coords[0]) is list or isinstance(self.coords[0], np.ndarray)
        ):
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
            self.coords_numpy = np.vstack((self.coords_numpy, self.coords_numpy[0]))
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
                f"Got wrong point {pt_coord}: Dimension more than 3rd provided."
            )
        if dim < 3:
            opt = np.lib.pad(opt, ((0, 3 - dim)), "constant", constant_values=0)
        return opt

    def new_pnt(self, pt_coords: list):
        pt_coords = self.pnt(pt_coords)
        for i, v in self.pts_index.items():
            if self.pnt_overlap(v, pt_coords):
                return False, i
        return True, None

    def pnt_overlap(self, pt1: np.ndarray, pt2: np.ndarray) -> bool:
        return np.isclose(np.linalg.norm(pt1 - pt2), 0)

    def init_pnts(self) -> None:
        for i, pt in enumerate(self.coords):
            pt_id = self.register_pnt(pt)
            if i != len(self.coords) - 1:
                self.init_pts_sequence.append(pt_id)
            if i != 0:
                self.update_digraph(self.init_pts_sequence[i - 1], pt_id)
                self.init_pts_sequence[i - 1] = [self.init_pts_sequence[i - 1], pt_id]

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

    def update_digraph(
        self,
        start_node: int,
        end_node: int,
        insert_node: int = None,
        build_new_edge: bool = True,
    ) -> None:
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
                end_node_list_index = self.pts_digraph[start_node].index(end_node)
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
        self.find_overlap_node_on_edge_parallel()
        self.modify_edge()

    def enclose(self):
        pass

    def get_segment(self, pt1: int, pt2: int) -> np.ndarray:
        return np.array([self.pts_index[pt1], self.pts_index[pt2]])

        # def init_segments(self):
        for i, v in enumerate(self.pts_sequance):
            if i != len(self.pts_sequance) - 1:
                self.segments_index.update({v: [self.pts_sequance[i + 1]]})
                self.count_vector_id += 1

    def insert_item(
        self, *items: np.ndarray, original: np.ndarray, insert_after: int
    ) -> np.ndarray:
        print(original[: insert_after + 1])
        return np.concatenate(
            (original[: insert_after + 1], items, original[insert_after + 1 :])
        )

    def add_pending_change(self, edge: tuple, new_node: int) -> None:
        if edge in self.modify_edge_list:
            self.modify_edge_list[edge].append(new_node)
        else:
            self.modify_edge_list.update({edge: [new_node]})

    def modify_edge(self):
        for edge, nodes in self.modify_edge_list.items():
            edge_0_coords = self.pts_index[edge[0]]
            nodes_coords = [self.pts_index[i] for i in nodes]
            distances = [np.linalg.norm(i - edge_0_coords) for i in nodes_coords]
            order = np.argsort(distances)
            nodes = [nodes[i] for i in order]
            self.pts_digraph[edge[0]].remove(edge[1])
            pts_list = [edge[0]] + nodes + [edge[1]]
            for i, nd in enumerate(pts_list):
                if i == 0:
                    continue
                self.update_digraph(pts_list[i - 1], nd, build_new_edge=False)
                if (i != 1) and (i != len(pts_list) - 1):
                    if (pts_list[i - 1] in self.virtual_pnt) and nd in (
                        self.virtual_pnt
                    ):
                        self.virtual_vector.update({(pts_list[i - 1], nd): True})

    def check_self_edge(self, line: np.ndarray) -> bool:
        if self.pnt_overlap(line[0], line[1]):
            return True
        else:
            return False

    def overlap_node_on_edge_finder(self, i, j):
        v = self.init_pts_sequence[i]
        vv = self.init_pts_sequence[j]
        print(i, j)
        lin1 = self.get_segment(v[0], v[1])
        lin2 = self.get_segment(vv[0], vv[1])
        self_edge = self.check_self_edge(lin1) or self.check_self_edge(lin2)
        if not self_edge:
            parallel, colinear = bcad.check_parallel_line_line(lin1, lin2)
            # if v == [13,14]:
            #     print("line:",(v,vv), parallel, colinear)
            if parallel:
                if colinear:
                    index, coords = bcad.check_overlap(lin1, lin2)
                    if len(index) < 4:
                        for ind in index:
                            if ind in [0, 1]:
                                self.add_pending_change(tuple(vv), v[ind])
                            else:
                                self.add_pending_change(tuple(v), vv[ind])
            else:
                distance = bcad.shortest_distance_line_line(lin1, lin2)
                intersect = np.isclose(distance[0], 0)
                new, pt = self.new_pnt(distance[1][0])
                if intersect and new:
                    pnt_id = self.register_pnt(distance[1][0])
                    self.add_pending_change(tuple(v), pnt_id)
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
                if (i, j) in visited or (j, i) in visited:
                    continue
                args = (self, i, j)
                print(args)
                yield args
            visited.update({(i, j): True, (j, i): True})

    # def find_overlap_node_on_edge(self):
    #     visited = {}
    #     for i, v in enumerate(self.init_pts_sequence):
    #         for j, vv in enumerate(self.init_pts_sequence):
    #             if i == j:
    #                 continue
    #             if i == j + 1:
    #                 continue
    #             if j == i + 1:
    #                 continue
    #             if i == len(self.init_pts_sequence) * 2 - 1 - i:
    #                 continue
    #             if (i, j) in visited or (j, i) in visited:
    #                 continue
    #             print(i, j)
    #             lin1 = self.get_segment(v[0], v[1])
    #             lin2 = self.get_segment(vv[0], vv[1])
    #             self_edge = self.check_self_edge(lin1) or self.check_self_edge(lin2)
    #             if not self_edge:
    #                 parallel, colinear = bcad.check_parallel_line_line(lin1, lin2)
    #                 # if v == [13,14]:
    #                 #     print("line:",(v,vv), parallel, colinear)
    #                 if parallel:
    #                     if colinear:
    #                         index, coords = bcad.check_overlap(lin1, lin2)
    #                         if len(index) < 4:
    #                             for ind in index:
    #                                 if ind in [0, 1]:
    #                                     self.add_pending_change(tuple(vv), v[ind])
    #                                 else:
    #                                     self.add_pending_change(tuple(v), vv[ind - 2])
    #                 else:
    #                     distance = bcad.shortest_distance_line_line(lin1, lin2)
    #                     intersect = np.isclose(distance[0], 0)
    #                     new, pt = self.new_pnt(distance[1][0])
    #                     if intersect and new:
    #                         pnt_id = self.register_pnt(distance[1][0])
    #                         self.virtual_pnt.update({pnt_id: True})
    #                         self.add_pending_change(tuple(v), pnt_id)
    #                         self.add_pending_change(tuple(vv), pnt_id)
    #             visited.update({(i, j): True, (j, i): True})
    
    def find_overlap_node_on_edge(self, line1, line2):
        print(line1, line2)
        lin1 = self.get_segment(line1[0], line1[1])
        lin2 = self.get_segment(line2[0], line2[1])
        self_edge = self.check_self_edge(lin1) or self.check_self_edge(lin2)
        if not self_edge:
            parallel, colinear = bcad.check_parallel_line_line(lin1, lin2)
            # if v == [13,14]:
            #     print("line:",(v,vv), parallel, colinear)
            if parallel:
                if colinear:
                    index, coords = bcad.check_overlap(lin1, lin2)
                    if len(index) < 4:
                        for ind in index:
                            if ind in [0, 1]:
                                self.add_pending_change(tuple(line2), line1[ind])
                            else:
                                self.add_pending_change(tuple(line1), line2[ind - 2])
            else:
                distance = bcad.shortest_distance_line_line(lin1, lin2)
                intersect = np.isclose(distance[0], 0)
                new, pt = self.new_pnt(distance[1][0])
                if intersect and new:
                    pnt_id = self.register_pnt(distance[1][0])
                    self.virtual_pnt.update({pnt_id: True})
                    self.add_pending_change(tuple(line1), pnt_id)
                    self.add_pending_change(tuple(line2), pnt_id)

    def find_overlap_node_on_edge_parallel(self):
        line_pairs = [
            (line1, line2)
            for line1 in self.init_pts_sequence
            for line2 in self.init_pts_sequence
            if line1 != line2
        ]
        num_processes = multiprocessing.cpu_count()
        chunk_size = len(line_pairs) // num_processes
        chunks = [
            line_pairs[i : i + chunk_size]
            for i in range(0, len(line_pairs), chunk_size)
        ]
        with multiprocessing.Pool(num_processes) as pool:
            results = pool.map(self.find_overlap_node_on_edge, chunks)


class CreateWallByPoints:
    def __init__(self, coords: list, th: float, height: float, is_close: bool = True):
        self.coords = Pnt(coords).coords
        self.height = height
        self.R = None
        self.interpolate = 8
        self.th = th
        self.is_close = is_close
        self.in_wall_pts_list = {}
        self.in_loop_pts_list = {}
        self.result_loops = []
        self.vecs = []
        self.central_segments = []
        self.dir_vecs = []
        self.ths = []
        self.lft_coords = []
        self.rgt_coords = []
        self.volume = 0
        self.side_coords: list
        self.create_sides()
        self.pnts = Segments(self.side_coords)
        self.G = nx.from_dict_of_lists(self.pnts.pts_digraph, create_using=nx.DiGraph)
        # self.all_loops = list(nx.simple_cycles(self.H)) # Dangerous! Ran out of memory.
        self.loop_generator = nx.simple_cycles(self.G)
        # self.check_pnt_in_wall()
        self.postprocessing()

    def create_sides(self):
        if self.R is not None:
            self.coords = bcad.interpolate_polygon(self.coords, self.interpolate)
            self.coords = bcad.bend(self.coords, self.R)
            self.coords = [i for i in self.coords]
        self.th *= 0.5
        for i, p in enumerate(self.coords):
            if i != len(self.coords) - 1:
                self.central_segments.append([self.coords[i], self.coords[i + 1]])
                a1 = self.coords[i + 1] - self.coords[i]
                if i == 0:
                    if self.is_close:
                        dr = bcad.bisect_angle(self.coords[-1] - p, a1)
                        # ang = angle_of_two_arrays(dir_vecs[i-1],dr)
                        ang2 = bcad.angle_of_two_arrays(
                            bcad.get_literal_vector(p - self.coords[-1], True), dr
                        )
                        ang_th = ang2
                        if ang2 > np.pi / 2:
                            dr *= -1
                            ang_th = np.pi - ang2
                        nth = np.abs(self.th / np.cos(ang_th))
                    else:
                        dr = bcad.get_literal_vector(a1, True)
                        nth = self.th
                else:
                    dr = bcad.bisect_angle(-self.vecs[i - 1], a1)
                    ang2 = bcad.angle_of_two_arrays(
                        bcad.get_literal_vector(self.vecs[i - 1], True), dr
                    )
                    ang_th = ang2
                    if ang2 > np.pi / 2:
                        dr *= -1
                        ang_th = np.pi - ang2
                    nth = np.abs(self.th / np.cos(ang_th))
            else:
                if self.is_close:
                    self.central_segments.append([self.coords[i], self.coords[0]])
                    a1 = self.coords[0] - self.coords[i]
                    dr = bcad.bisect_angle(-self.vecs[i - 1], a1)
                    ang2 = bcad.angle_of_two_arrays(
                        bcad.get_literal_vector(self.vecs[i - 1], True), dr
                    )
                    ang_th = ang2
                    if ang2 > np.pi / 2:
                        dr *= -1
                        ang_th = np.pi - ang2
                    nth = np.abs(self.th / np.cos(ang_th))
                else:
                    dr = bcad.get_literal_vector(a1, True)
                    nth = self.th
            self.vecs.append(a1)
            self.ths.append(nth)
            self.dir_vecs.append(dr)
            self.lft_coords.append(dr * nth + p)
            self.rgt_coords.append(-dr * nth + p)
        if self.is_close:
            self.lft_coords.append(self.lft_coords[0])
            self.rgt_coords.append(self.rgt_coords[0])
            self.rgt_coords = self.rgt_coords[::-1]
            self.coords.append(self.coords[0])
            self.side_coords = self.lft_coords + self.rgt_coords
        else:
            self.rgt_coords = self.rgt_coords[::-1]
            self.side_coords = self.lft_coords + self.rgt_coords + [self.lft_coords[0]]

    def check_pnt_in_wall(self):
        for pnt, coord in self.pnts.pts_index.items():
            for vec in self.central_segments:
                lmbda, dist = bcad.shortest_distance_point_line(vec, coord)
                if dist < 0.9 * self.th:
                    print(f"pnt:{pnt},dist:{dist},lmbda:{lmbda}, vec:{vec}")
                    self.in_wall_pts_list.update({pnt: True})
                    break

    def visualize(
        self,
        display_polygon: bool = True,
        display_central_path: bool = False,
        all_polygons: bool = False,
    ):
        # Extract the x and y coordinates and IDs
        a = self.pnts.pts_index
        x = [coord[0] for coord in a.values()]
        y = [coord[1] for coord in a.values()]
        ids = list(a.keys())  # Get the point IDs
        # Create a scatter plot in 2D
        plt.subplot(1, 2, 1)
        # plt.figure()
        plt.scatter(x, y)

        # Annotate points with IDs
        for i, (xi, yi) in enumerate(zip(x, y)):
            plt.annotate(f"{ids[i]}", (xi, yi), fontsize=12, ha="right")

        if display_polygon:
            if all_polygons:
                display_loops = nx.simple_cycles(self.G)
            else:
                display_loops = self.result_loops
            for lp in display_loops:
                coords = [self.pnts.pts_index[i] for i in lp]
                x = [point[0] for point in coords]
                y = [point[1] for point in coords]
                plt.plot(x + [x[0]], y + [y[0]], linestyle="-", marker="o")
        if display_central_path:
            talist = np.array(self.coords).T
            x1 = talist[0]
            y1 = talist[1]
            plt.plot(x1, y1, "bo-", label="central path", color="b")

        # Create segments by connecting consecutive points

        # a_subtitute = np.array(self.side_coords)
        # toutput1 = a_subtitute.T
        # x2 = toutput1[0]
        # y2 = toutput1[1]
        # plt.plot(x2, y2, 'ro-', label='outer line')

        # Set labels and title
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.title("Points With Polygons detected")

        plt.subplot(1, 2, 2)
        # layout = nx.spring_layout(self.G)
        layout = nx.circular_layout(self.G)
        # Draw the nodes and edges
        nx.draw(
            self.G,
            pos=layout,
            with_labels=True,
            node_color="skyblue",
            font_size=10,
            node_size=300,
        )
        plt.title("Multi-Digraph")
        plt.tight_layout()
        # Show the plot
        plt.show()

    def get_loops(self):
        return [i for i in self.all_loops if len(i) > 2]

    def visualize_graph(self):
        layout = nx.spring_layout(self.G)
        # Draw the nodes and edges
        nx.draw(
            self.G,
            pos=layout,
            with_labels=True,
            node_color="skyblue",
            font_size=10,
            node_size=500,
        )
        plt.title("NetworkX Graph Visualization")
        plt.show()

    def Shape(self):
        loop_r = self.rank_result_loops()
        print(loop_r)
        boundary = [self.occ_pnt(self.pnts.pts_index[i]) for i in loop_r[0]]
        poly0 = bcad.random_polygon_constructor(boundary)
        poly_r = poly0
        for i, h in enumerate(loop_r):
            if i == 0:
                continue
            h = [self.occ_pnt(self.pnts.pts_index[i]) for i in h]
            poly_c = bcad.random_polygon_constructor(h)
            poly_r = occh.cut(poly_r, poly_c)
        self.poly = poly_r
        if not np.isclose(self.height, 0):
            wall_compound = occh.create_prism(self.poly, [0, 0, self.height])
            faces = occh.explore_topo(wall_compound, "face")
            wall_shell = occh.sew_face(faces)
            self.wall = occh.create_solid(wall_shell)
            return self.wall
        else:
            return self.poly

    def postprocessing(self):
        correct_loop_count = 0
        for lp in self.loop_generator:
            real_loop = True
            visible_loop = True
            in_wall_pt_count = 0
            virtual_vector_count = 0
            if len(lp) > 2:
                for i, pt in enumerate(lp):
                    if i == 0:
                        if pt in self.in_wall_pts_list:
                            in_wall_pt_count += 1
                        if (lp[-1], pt) in self.pnts.virtual_vector:
                            virtual_vector_count += 1
                    else:
                        if pt in self.in_wall_pts_list:
                            in_wall_pt_count += 1
                        if (lp[i - 1], pt) in self.pnts.virtual_vector:
                            virtual_vector_count += 1
                    if (
                        (in_wall_pt_count > 0)
                        or (virtual_vector_count > 1)
                        # or ((in_wall_pt_count == 0) and (virtual_vector_count > 0))
                    ):
                        # if (in_wall_pt_count > 0):
                        visible_loop = False
                        break
            else:
                real_loop = False
            if real_loop and visible_loop:
                self.result_loops.append(lp)
                for pt in lp:
                    if pt not in self.in_loop_pts_list:
                        self.in_loop_pts_list.update({pt: [correct_loop_count]})
                    else:
                        self.in_loop_pts_list[pt].append(correct_loop_count)
                correct_loop_count += 1
        loop_counter = np.zeros(len(self.result_loops))
        visited = []
        counter = 0
        for pt, lp in self.in_loop_pts_list.items():
            if len(lp) > 1:
                if counter == 0:
                    visited.append(lp)
                    for ind in lp:
                        loop_counter[ind] += 1
                else:
                    if any(sorted(lp) == sorted(item) for item in visited):
                        continue
                    else:
                        visited.append(lp)
                        for ind in lp:
                            loop_counter[ind] += 1
                counter += 1
        filtered_lp = np.where(loop_counter > 1)[0].tolist()
        print("filtered:", filtered_lp)
        print("vote:", loop_counter)
        self.result_loops = [
            v for i, v in enumerate(self.result_loops) if i not in filtered_lp
        ]
        print("result:", self.result_loops)

    def rank_result_loops(self):
        areas = np.zeros(len(self.result_loops))
        for i, lp in enumerate(self.result_loops):
            lp_coord = [self.pnts.pts_index[i] for i in lp]
            area = bcad.get_face_area(lp_coord)
            areas[i] = area
        rank = np.argsort(areas).tolist()
        self.volume = (2 * np.max(areas) - np.sum(areas)) * self.height * 1e-6
        self.result_loops = sorted(
            self.result_loops,
            key=lambda x: rank.index(self.result_loops.index(x)),
            reverse=True,
        )
        return self.result_loops

    def occ_pnt(self, coord) -> gp_Pnt:
        return gp_Pnt(*coord)
