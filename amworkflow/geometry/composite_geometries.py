import amworkflow.geometry.builtinCAD as bcad
import numpy as np
from pprint import pprint


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
                {"CWBP": {"center_point": True, "order": i, "derive": [None, None]}}
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

        # if self.is_close:
        #     self.lft_coords.append(self.lft_coords[0])
        #     self.rgt_coords.append(self.rgt_coords[0])
        #     self.rgt_coords = self.rgt_coords[::-1]
        #     self.coords.append(self.coords[0])
        #     self.side_coords = self.lft_coords + self.rgt_coords
        # else:
        #     self.rgt_coords = self.rgt_coords[::-1]
        #     self.side_coords = self.lft_coords + self.rgt_coords + [self.lft_coords[0]]

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

# pnt1 = bcad.Pnt([2, 3])
# pnt2 = bcad.Pnt([1, 5])
# pnt3 = bcad.Pnt([2, 4])
# pnt4 = bcad.Pnt([2, 9])
# seg1 = bcad.Segment(pnt1, pnt2)
# seg2 = bcad.Segment(pnt2, pnt3)
# seg3 = bcad.Segment(pnt3, pnt1)
# # wire1 = bcad.Wire(seg1, seg2,seg3)
# # surf1 = bcad.Surface(wire1)
# # pprint(bcad.id_index)
# # print(seg3)
# # print(pnt1.property["occ_vector"])
# # pnts_handler = PntHandler()
# # pnts_handler.init_center_points(pnt1, pnt2, pnt3)
# # pnts_handler.handle_boundary_point(pnt1, pnt4)
# # pprint(bcad.id_index)
# segments = SegmentHandler(pnt1, pnt2, pnt3, thickness=0.5, is_close=True)
# print(segments.center_line)
