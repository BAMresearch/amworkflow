from pprint import pprint

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from OCC.Core.gp import gp_Dir, gp_Pnt, gp_Vec

import amworkflow.geometry.builtinCAD as bcad
from amworkflow import occ_helpers as occh
from amworkflow.geometry import simple_geometries as sgeom


class CreateWallByPoints:
    """
    Create a wall by points.
    Parts of this class consists of PntHandler and SegmentHandler, which handles processing data of points and segments.
    """

    def __init__(
        self, *pnts, thickness: float, height: float = 0, radius: float = None
    ) -> None:
        self.segments = SegmentHandler(*pnts)
        self.R = radius
        self.hth = thickness
        self.h = height


class PntHandler:
    """
    Handle the points of the wall.
    """

    def __init__(self) -> None:
        self.pnts = []
        self.pnt_ids = []
        self.pnt_coords = []
        self.pnt_property = {}

    def init_pnts(self, *pnts) -> None:
        self.pnts.extend(pnts)
        self.pnt_ids.extend([item.id for item in pnts])
        self.pnt_coords.extend([item.value for item in pnts])
        # self.pnt_property.update({key:item for key, item in bcad.id_index.items() if item["type"] == 0 and item["id"] in self.pnt_ids})

    def init_center_points(self, *pnts) -> None:
        self.init_pnts(*pnts)
        for i, pnt in enumerate(pnts):
            pnt.enrich_property(
                {
                    "CWBP": {
                        "center_point": True,
                        "order": i,
                        "derive": [None, None, None],
                    }
                }
            )

    def init_boundary_point(self, pnt: bcad.Pnt, is_left: bool) -> None:
        self.init_pnts(pnt)
        pnt.enrich_property(
            {"CWBP": {"center_point": False, "literality": is_left, "originate": None}}
        )
        return pnt

    def get_pnt_coord(self, pnt_id):
        if isinstance(pnt_id, bcad.Pnt):
            pnt_id = pnt_id.id
        if pnt_id not in self.pnt_ids:
            raise Exception(f"Unrecognized point id: {pnt_id}.")
        return bcad.id_index[pnt_id]["value"]

    def handle_boundary_point(
        self, center_point: bcad.Pnt, boundary_point: bcad.Pnt
    ) -> None:
        if boundary_point.property["CWBP"]["center_point"]:
            raise Exception(f"Unrecognized boundary point: {boundary_point.id}.")
        if not center_point.property["CWBP"]["center_point"]:
            raise Exception(f"Unrecognized center point: {center_point.id}.")
        if boundary_point.property["CWBP"]["originate"] is not None:
            raise Exception(f"Boundary point {boundary_point.id} has been handled.")
        center_pnt_left_handled = center_point.property["CWBP"]["derive"][0] is not None
        center_pnt_rgt_handled = center_point.property["CWBP"]["derive"][1] is not None
        if center_pnt_left_handled and center_pnt_rgt_handled:
            raise Exception(f"Center point {center_point.id} has been handled.")
        if boundary_point.property["CWBP"]["literality"]:
            center_point.property["CWBP"]["derive"][0] = boundary_point.id
        else:
            center_point.property["CWBP"]["derive"][1] = boundary_point.id
        boundary_point.enrich_property({"CWBP": {"center_point": False}})
        boundary_point.property["CWBP"].update({"originate": center_point.id})


class SegmentHandler:
    """
    Handle the segments of the wall.
    """

    def __init__(self, *pnts: bcad.Pnt, thickness: float, is_close: bool) -> None:
        super().__init__()
        self.hth = 0.5 * thickness
        self.center_line = []
        self.support_vectors = []
        self.digraph = {}
        self.is_close = is_close
        self.pnt_handler = PntHandler()
        self.pnt_handler.init_center_points(*pnts)
        self.pnt_ids = self.pnt_handler.pnt_ids
        self.coords = self.pnt_handler.pnt_coords
        self.lft_side_pnts = []
        self.init_center_line()
        self.init_boundary()

    def init_center_line(self) -> None:
        """
        Initialize the center line of the wall.
        """
        CORNER_COMPENSATION_THRESHOLD = np.pi / 6
        center_pnts = [
            item
            for item in self.pnt_handler.pnts
            if bcad.id_index[item.id]["type"] == 0
            and bcad.id_index[item.id]["CWBP"]["center_point"]
        ]
        lst_pnt_coord = self.pnt_handler.get_pnt_coord(center_pnts[-1])
        for i, pt in enumerate(center_pnts):
            pt_coord = self.pnt_handler.get_pnt_coord(pt)
            # nxt_pnt_coord = self.pnt_handler.get_pnt_coord(center_pnts[i+1])
            if i != len(center_pnts) - 1:
                c_vector = bcad.Segment(pt, center_pnts[i + 1]).vector
                if i == 0:
                    if self.is_close:
                        support_vector = bcad.Segment(
                            bcad.Pnt(
                                bcad.angular_bisector(
                                    lst_pnt_coord - pt_coord, c_vector
                                )
                            )
                        ).vector
                        # ang = angle_of_two_arrays(dir_vecs[i-1],support_vector)
                        ang2 = bcad.angle_of_two_arrays(
                            bcad.laterality_indicator(pt_coord - lst_pnt_coord, True),
                            support_vector,
                        )
                        ang3 = bcad.angle_of_two_arrays(
                            c_vector, lst_pnt_coord - pt_coord
                        )
                        ang_th = ang2
                        if ang2 > np.pi / 2:
                            support_vector *= -1
                            ang_th = np.pi - ang2
                        nth = np.abs(self.hth / np.cos(ang_th))
                    else:
                        support_vector = bcad.Segment(
                            bcad.Pnt(bcad.laterality_indicator(c_vector, True))
                        ).vector
                        nth = self.hth
                else:
                    support_vector = bcad.Segment(
                        bcad.Pnt(
                            bcad.angular_bisector(-self.center_line[i - 1], c_vector)
                        )
                    ).vector
                    ang2 = bcad.angle_of_two_arrays(
                        bcad.laterality_indicator(self.center_line[i - 1], True),
                        support_vector,
                    )
                    ang3 = bcad.angle_of_two_arrays(c_vector, self.center_line[i - 1])
                    ang_th = ang2
                    if ang2 > np.pi / 2:
                        support_vector *= -1
                        ang_th = np.pi - ang2
                    nth = np.abs(self.hth / np.cos(ang_th))
            else:
                if self.is_close:
                    c_vector = self.coords[0] - self.coords[i]
                    support_vector = bcad.Segment(
                        bcad.Pnt(
                            bcad.angular_bisector(-self.center_line[i - 1], c_vector)
                        )
                    ).vector
                    ang2 = bcad.angle_of_two_arrays(
                        bcad.laterality_indicator(self.center_line[i - 1], True),
                        support_vector,
                    )
                    ang3 = bcad.angle_of_two_arrays(c_vector, self.center_line[i - 1])
                    ang_th = ang2
                    if ang2 > np.pi / 2:
                        support_vector *= -1
                        ang_th = np.pi - ang2
                    nth = np.abs(self.hth / np.cos(ang_th))
                else:
                    support_vector = bcad.Segment(
                        bcad.Pnt(bcad.laterality_indicator(c_vector, True))
                    ).vector
                    nth = self.hth
            self.center_line.append(c_vector)
            self.support_vectors.append(support_vector)
            if ang3 > CORNER_COMPENSATION_THRESHOLD:
                # On the left side of the center line.
                if ang2 > np.pi / 2:
                    pass
            lft_pnt = self.pnt_handler.init_boundary_point(
                bcad.Pnt(pt_coord + support_vector * nth), True
            )
            rgt_pnt = self.pnt_handler.init_boundary_point(
                bcad.Pnt(pt_coord - support_vector * nth), False
            )
            self.pnt_handler.handle_boundary_point(pt, lft_pnt)
            self.pnt_handler.handle_boundary_point(pt, rgt_pnt)

    def init_boundary(self) -> None:
        """
        Initialize the boundary of the wall.
        """

        if self.is_close:
            self.lft_coords.append(self.lft_coords[0])
            self.rgt_coords.append(self.rgt_coords[0])
            self.rgt_coords = self.rgt_coords[::-1]
            self.coords.append(self.coords[0])
            self.side_coords = self.lft_coords + self.rgt_coords
        else:
            self.rgt_coords = self.rgt_coords[::-1]
            self.side_coords = self.lft_coords + self.rgt_coords + [self.lft_coords[0]]

    def update_digraph(
        self,
        start_node: int,
        end_node: int,
        insert_node: int = None,
        build_new_edge: bool = True,
    ) -> None:
        """
        Update the digraph of the points.
        """
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
                end_node_list_index = self.digraph[start_node].index(end_node)
                self.digraph[start_node][end_node_list_index] = insert_node
                if build_new_edge:
                    self.digraph.update({insert_node: [end_node]})
        else:
            if insert_node is None:
                self.digraph.update({start_node: [end_node]})
            else:
                raise Exception("No edge found for insertion option.")

    def calculate_boundary(self) -> None:
        """
        Calculate the boundary of the wall.
        """


class CreateWallByPointsUpdate:
    def __init__(self, coords: list, th: float, height: float, is_close: bool = True):
        self.coords = sgeom.Pnt(coords).coords
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
        self.side_coords: list
        self.create_sides()
        self.pnts = sgeom.Segments(self.side_coords)
        self.G = nx.from_dict_of_lists(self.pnts.pts_digraph, create_using=nx.DiGraph)
        # self.all_loops = list(nx.simple_cycles(self.H))  # Dangerous! Ran out of memory.
        self.loop_generator = nx.simple_cycles(self.G)
        self.check_pnt_in_wall()
        self.postprocessing()

    def create_sides(self):
        if self.R is not None:
            self.coords = bcad.polygon_interpolater(self.coords, self.interpolate)
            self.coords = bcad.bender(self.coords, self.R)
            self.coords = [i for i in self.coords]
        self.th *= 0.5
        for i, p in enumerate(self.coords):
            if i != len(self.coords) - 1:
                self.central_segments.append([self.coords[i], self.coords[i + 1]])
                a1 = self.coords[i + 1] - self.coords[i]
                if i == 0:
                    if self.is_close:
                        dr = bcad.angular_bisector(self.coords[-1] - p, a1)
                        # ang = angle_of_two_arrays(dir_vecs[i-1],dr)
                        ang2 = bcad.angle_of_two_arrays(
                            bcad.laterality_indicator(p - self.coords[-1], True), dr
                        )
                        ang_th = ang2
                        if ang2 > np.pi / 2:
                            dr *= -1
                            ang_th = np.pi - ang2
                        nth = np.abs(self.th / np.cos(ang_th))
                    else:
                        dr = bcad.laterality_indicator(a1, True)
                        nth = self.th
                else:
                    dr = bcad.angular_bisector(-self.vecs[i - 1], a1)
                    ang2 = bcad.angle_of_two_arrays(
                        bcad.laterality_indicator(self.vecs[i - 1], True), dr
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
                    dr = bcad.angular_bisector(-self.vecs[i - 1], a1)
                    ang2 = bcad.angle_of_two_arrays(
                        bcad.laterality_indicator(self.vecs[i - 1], True), dr
                    )
                    ang_th = ang2
                    if ang2 > np.pi / 2:
                        dr *= -1
                        ang_th = np.pi - ang2
                    nth = np.abs(self.th / np.cos(ang_th))
                else:
                    dr = bcad.laterality_indicator(a1, True)
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
        poly0 = bcad.create_polygon_by_points(boundary)
        poly_r = poly0
        for i, h in enumerate(loop_r):
            if i == 0:
                continue
            h = [self.occ_pnt(self.pnts.pts_index[i]) for i in h]
            poly_c = bcad.create_polygon_by_points(h)
            poly_r = occh.cutter3D(poly_r, poly_c)
        self.poly = poly_r
        if not np.isclose(self.height, 0):
            wall_compound = sgeom.create_prism(self.poly, [0, 0, self.height])
            faces = occh.topo_explorer(wall_compound, "face")
            wall_shell = occh.sew_face(faces)
            self.wall = sgeom.create_solid(wall_shell)
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
                        or ((in_wall_pt_count == 0) and (virtual_vector_count > 0))
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
        self.result_loops = sorted(
            self.result_loops,
            key=lambda x: rank.index(self.result_loops.index(x)),
            reverse=True,
        )
        return self.result_loops

    def occ_pnt(self, coord) -> gp_Pnt:
        return gp_Pnt(*coord)


# pnt1 = bcad.Pnt([2,3])
# pnt2 = bcad.Pnt([1,5])
# pnt3 = bcad.Pnt([2,4])
# pnt4 = bcad.Pnt([2,9])
# seg1 = bcad.Segment(pnt1, pnt2)
# seg2 = bcad.Segment(pnt2, pnt3)
# seg3 = bcad.Segment(pnt3, pnt1)
# wire1 = bcad.Wire(seg1, seg2,seg3)
# surf1 = bcad.Surface(wire1)
# pprint(bcad.id_index)
# print(seg3)
# print(pnt1.property["occ_vector"])
# pnts_handler = PntHandler()
# pnts_handler.init_center_points(pnt1, pnt2, pnt3)
# pnts_handler.handle_boundary_point(pnt1, pnt4)
# pprint(bcad.id_index)
# segments = SegmentHandler(pnt1, pnt2, pnt3, thickness=0.5, is_close=True)
# print(segments.center_line)
