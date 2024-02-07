import logging
from pprint import pprint
from typing import Union

import numpy as np
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakePolygon
from OCC.Core.gp import gp_Pnt
from OCC.Core.TopoDS import (
    TopoDS_Compound,
    TopoDS_Edge,
    TopoDS_Face,
    TopoDS_Shape,
    TopoDS_Shell,
    TopoDS_Solid,
    TopoDS_Wire,
)

from amworkflow.geometry.simple_geometries import create_edge, create_face, create_wire
from amworkflow.occ_helpers import create_solid, sew_face

level = logging.WARNING
logging.basicConfig(
    level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("amworkflow.geometry.builtinCAD")
# logger.setLevel(logging.INFO)

count_id = 0
count_gid = [0 for i in range(7)]
id_index = {}
TYPE_INDEX = {
    0: "point",
    1: "segment",
    2: "wire",
    3: "surface",
    4: "shell",
    5: "solid",
    6: "compound",
}


def pnt(pt_coord) -> np.ndarray:
    """
    Create a point.
    :param pt_coord: The coordinate of the point. If the dimension is less than 3, the rest will be padded with 0. If the dimension is more than 3, an exception will be raised.
    :type pt_coord: list
    :return: The coordinate of the point.
    :rtype: np.ndarray
    """
    opt = np.array(pt_coord)
    dim = np.shape(pt_coord)[0]
    if dim > 3:
        raise Exception(
            f"Got wrong point {pt_coord}: Dimension more than 3rd provided."
        )
    if dim < 3:
        opt = np.lib.pad(opt, ((0, 3 - dim)), "constant", constant_values=0)
    return opt


class DuplicationCheck:
    """
    Check if an item already exists in the index.
    """

    def __init__(
        self,
        gtype: int,
        gvalue: any,
    ) -> None:
        self.gtype = gtype
        self.gvalue = gvalue
        self.check_type_validity(item_type=self.gtype)
        self.new, self.exist_object = self.new_item(gvalue, gtype)
        if not self.new:
            self.exist_object.re_init = True
            logging.info(
                f"{TYPE_INDEX[gtype]} {gvalue} already exists, return the old one."
            )

    def check_type_coincide(self, base_type: int, item_type: int) -> bool:
        """Check if items has the same type with the base item.

        :param base_type: The type referred
        :type base_type: int
        :param item_type: The type to be examined
        :type item_type: int
        :return: True if coincident.
        :rtype: bool
        """
        different = base_type != item_type
        self.check_type_validity(item_type=item_type)
        if different:
            return False
        else:
            return True

    def check_type_validity(self, item_type: int) -> bool:
        """Check if given type if valid

        :param item_type: The type to be examined
        :type item_type: int
        :raises Exception: Wrong geometry object type, perhaps a mistake made in development.
        :return: True if valid
        :rtype: bool
        """
        valid = item_type in TYPE_INDEX
        if not valid:
            raise Exception(
                "Wrong geometry object type, perhaps a mistake made in development."
            )
        return valid

    def check_value_repetition(self, base_value: any, item_value: any) -> bool:
        """
        Check if a value is close enough to the base value. True if repeated.
        :param base_value: item to be compared with.
        :param item_value: value to be examined.
        """
        if type(base_value) is list:
            base_value = np.array(base_value)
            item_value = np.array(item_value)

        return np.isclose(np.linalg.norm(base_value - item_value), 0)

    def new_item(self, item_value: any, item_type) -> tuple:
        """
        Check if a value already exits in the index
        :param item_value: value to be examined.
        """
        for _, item in id_index.items():
            if self.check_type_coincide(item["type"], item_type):
                if self.check_value_repetition(item["value"], item_value):
                    return False, item[f"{TYPE_INDEX[item_type]}"]
        return True, None


class TopoObj:
    def __init__(self) -> None:
        """
        TopoObj
        ----------
        The Base class for all builtin Topo class.

        Geometry object type:
        0: point
        1: segment
        2: wire
        3: surface
        4: shell
        5: solid
        6: compound

        id: An unique identity number for every instance of topo_class
        gid: An unique identity number for every instance of topo_class with the same type

        """
        self.type = 0
        self.value = 0
        self.id = 0
        self.gid = 0
        self.own = {}
        self.belong = {}
        self.property = {}
        self.property_enriched = False

    def __str__(self) -> str:
        own = ""
        for item_type, item_value in self.own.items():
            item_type = TYPE_INDEX[item_type]
            for index, item_id in enumerate(item_value):
                if index != len(item_value) - 1:
                    own += f"{item_id}({item_type}),"
                else:
                    own += f"{item_id}({item_type})"
        belong = ""
        for item_type, item_value in self.belong.items():
            item_type = TYPE_INDEX[item_type]
            for index, item_id in enumerate(item_value):
                if index != len(item_value) - 1:
                    belong += f"{item_id}({item_type}),"
                else:
                    belong += f"{item_id}({item_type})"
        if self.type == 0:
            value = str(self.value) + "(coordinate)"
        else:
            value = str(self.value) + "(IDs)"
        doc = f"\033[1mType\033[0m: {TYPE_INDEX[self.type]}\n\033[1mID\033[0m: {self.id}\n\033[1mValue\033[0m: {value}\n\033[1mOwn\033[0m: {own}\n\033[1mBelong\033[0m: {belong}\n"
        return doc

    def enrich_property(self, new_property: dict):
        """Enrich the property out of the basic property.

        :param new_property: A dictionary containing new properties and their values.
        :type new_property: dict
        :raises Exception: New properties override existing properties.
        """
        if new_property in self.property.items():
            raise Exception("New properties override existing properties.")
        self.property.update(new_property)
        self.property_enriched = True

    def update_basic_property(self):
        """Update basic properties"""
        self.property.update(
            {
                "type": self.type,
                f"{TYPE_INDEX[self.type]}": self,
                "id": self.id,
                "gid": self.gid,
                "own": self.own,
                "belong": self.belong,
                "value": self.value,
            }
        )

    def update_property(self, property_key: str, property_value: any):
        """Update a property of the item.

        :param property_key: The key of the property to be updated.
        :type property_key: str
        :param property_value: The value of the property to be updated.
        :type property_value: any
        """
        if property_key not in self.property:
            raise Exception(f"Unrecognized property key: {property_key}.")
        self.property.update({property_key: property_value})

    def update_id_index(self):
        id_index[self.id].update(self.property)

    def register_item(self) -> int:
        """
        Register an item to the index and return its id. Duplicate value will be filtered.
        :param item_value: value to be registered.
        """
        # new, old_id = self.new_item(self.value, self.type)
        global count_id
        global count_gid
        # if new:
        self.id = count_id
        self.gid = count_gid[self.type]
        count_gid[self.type] += 1
        self.update_basic_property()
        id_index.update({self.id: self.property})
        count_id += 1
        return self.id
        # else:
        #     return old_id

    def update_dependency(self, *own: list):
        for item in own:
            if item.type in self.own:
                if item.id not in self.own:
                    self.own[item.type].append(item.id)
            else:
                self.own.update({item.type: [item.id]})
            if self.type in item.belong:
                if self.id not in item.belong[self.type]:
                    item.belong[self.type].append(self.id)
            else:
                item.belong.update({self.type: [self.id]})
        # self.update_basic_property()
        # self.update_id_index()


class Pnt(TopoObj):
    """
    Create a point.
    """

    def __new__(cls, coord: list = []) -> None:
        point = pnt(coord)
        checker = DuplicationCheck(0, point)
        if checker.new:
            instance = super().__new__(cls)
            instance.re_init = False
            return instance
        else:
            return checker.exist_object

    def __init__(self, coord: list = []) -> None:
        if self.re_init:
            return
        super().__init__()
        self.re_init = False
        self.type = 0
        self.coord = pnt(coord)
        self.value = self.coord
        self.occ_pnt = gp_Pnt(*self.coord.tolist())
        self.enrich_property({"occ_pnt": self.occ_pnt})
        self.id = self.register_item()


class Segment(TopoObj):
    def __new__(cls, pnt2: Pnt = Pnt([1]), pnt1: Pnt = Pnt([])):
        checker = DuplicationCheck(1, [pnt1.id, pnt2.id])
        if checker.new:
            instance = super().__new__(cls)
            instance.re_init = False
            return instance
        else:
            return checker.exist_object

    def __init__(self, pnt2: Pnt, pnt1: Pnt = Pnt([])) -> None:
        super().__init__()
        if self.re_init:
            return
        self.re_init = False
        pnt1, pnt2 = self.check_input(pnt1=pnt1, pnt2=pnt2)
        self.start_pnt = pnt1.id
        self.end_pnt = pnt2.id
        self.vector = pnt2.value - pnt1.value
        self.length = np.linalg.norm(self.vector)
        self.normal = self.vector / self.length
        self.type = 1
        self.value = [self.start_pnt, self.end_pnt]
        self.raw_value = [
            id_index[self.start_pnt]["value"],
            id_index[self.end_pnt]["value"],
        ]
        self.occ_edge = create_edge(pnt1.occ_pnt, pnt2.occ_pnt)
        self.enrich_property(
            {
                "occ_edge": self.occ_edge,
                "vector": self.vector,
                "length": self.length,
                "normal": self.normal,
            }
        )
        self.register_item()
        self.update_dependency(pnt1, pnt2)

    def check_input(self, pnt1: Pnt, pnt2: Pnt) -> tuple:
        if type(pnt2) is int:
            if pnt2 not in id_index:
                raise Exception(f"Unrecognized point id: {pnt2}.")
            pnt2 = id_index[pnt2]["point"]
        if type(pnt1) is int:
            if pnt1 not in id_index:
                raise Exception(f"Unrecognized point id: {pnt1}.")
            pnt1 = id_index[pnt1]["point"]
        if not isinstance(pnt1, Pnt):
            raise Exception(f"Wrong type of point: {type(pnt1)}.")
        if not isinstance(pnt2, Pnt):
            raise Exception(f"Wrong type of point: {type(pnt2)}.")
        if pnt1.id == pnt2.id:
            raise Exception(f"Start point and end point are the same: {pnt1.id}.")
        return pnt1, pnt2


class Wire(TopoObj):
    def __new__(cls, *segments: Segment):
        checker = DuplicationCheck(2, [item.id for item in segments])
        if checker.new:
            instance = super().__new__(cls)
            instance.re_init = False
            return instance
        else:
            return checker.exist_object

    def __init__(self, *segments: Segment) -> None:
        super().__init__()
        if self.re_init:
            return
        self.re_init = False
        self.type = 2
        self.seg_ids = [item.id for item in segments]
        self.occ_wire = create_wire(*[item.occ_edge for item in segments])
        self.update_dependency(*segments)
        self.value = self.seg_ids
        self.enrich_property({"occ_wire": self.occ_wire})
        self.register_item()


class Surface(TopoObj):
    def __new__(cls, *wire: Wire):
        checker = DuplicationCheck(3, [item.id for item in wire])
        if checker.new:
            instance = super().__new__(cls)
            instance.re_init = False
            return instance
        else:
            return checker.exist_object

    def __init__(self, *wires: Wire) -> None:
        super().__init__()
        if self.re_init:
            return
        self.re_init = False
        self.type = 3
        self.wire_ids = [item.id for item in wires]
        self.value = self.wire_ids
        self.occ_face = create_face(wires[0].occ_wire)
        self.update_dependency(*wires)
        self.enrich_property({"occ_face": self.occ_face})
        self.register_item()


class Shell(TopoObj):
    def __new__(cls, *surfaces: Surface):
        checker = DuplicationCheck(4, [item.id for item in surfaces])
        if checker.new:
            instance = super().__new__(cls)
            instance.re_init = False
            return instance
        else:
            return checker.exist_object

    def __init__(self, *surfaces: Surface) -> None:
        super().__init__()
        if self.re_init:
            return
        self.re_init = False
        self.type = 4
        self.surf_ids = [item.id for item in surfaces]
        self.value = self.surf_ids
        self.occ_shell = sew_face(*[item.occ_face for item in surfaces])
        self.update_dependency(*surfaces)
        self.enrich_property({"occ_shell": self.occ_shell})
        self.register_item()


class Solid(TopoObj):
    def __new__(cls, shell: Shell):
        checker = DuplicationCheck(5, shell.id)
        if checker.new:
            instance = super().__new__(cls)
            instance.re_init = False
            return instance
        else:
            return checker.exist_object

    def __init__(self, shell: Shell) -> None:
        super().__init__()
        if self.re_init:
            return
        self.re_init = False
        self.type = 5
        self.shell_id = shell.id
        self.value = [self.shell_id]
        self.occ_solid = create_solid(shell.occ_shell)
        self.update_dependency(shell)
        self.enrich_property({"occ_solid": self.occ_solid})
        self.register_item()


def create_wire_by_points(points: list):
    """
    @brief Create a closed wire (loop) by points. The wire is defined by a list of points which are connected by an edge.
    @param points A list of points. Each point is a gp_Pnt ( x y z) where x, y and z are the coordinates of a point.
    @return A wire with the given points connected by an edge. This will be an instance of : class : `BRepBuilderAPI_MakeWire`
    """
    pts = points
    # Create a wire for each point in the list of points.
    for i, pt in enumerate(pts):
        # Create a wire for the i th point.
        if i == 0:
            edge = create_edge(pt, pts[i + 1])
            wire = create_wire(edge)
        # Create a wire for the given points.
        if i != len(pts) - 1:
            edge = create_edge(pt, pts[i + 1])
            wire = create_wire(wire, edge)
        else:
            edge = create_edge(pts[i], pts[0])
            wire = create_wire(wire, edge)
    return wire


def random_polygon_constructor(
    points: list, isface: bool = True
) -> TopoDS_Face or TopoDS_Wire:
    """
    @brief Creates a polygon in any shape. If isface is True the polygon is made face - oriented otherwise it is wires
    @param points List of points defining the polygon
    @param isface True if you want to create a face - oriented
    @return A polygon
    """
    pb = BRepBuilderAPI_MakePolygon()
    # Add points to the points.
    for pt in points:
        pb.Add(pt)
    pb.Build()
    pb.Close()
    # Create a face or a wire.
    if isface:
        return create_face(pb.Wire())
    else:
        return pb.Wire()


def angle_of_two_arrays(a1: np.ndarray, a2: np.ndarray, rad: bool = True) -> float:
    """
    @brief Returns the angle between two vectors. This is useful for calculating the rotation angle between a vector and another vector
    @param a1 1D array of shape ( n_features )
    @param a2 2D array of shape ( n_features )
    @param rad If True the angle is in radians otherwise in degrees
    @return Angle between a1 and a2 in degrees or radians depending on rad = True or False
    """
    dot = np.dot(a1, a2)
    norm = np.linalg.norm(a1) * np.linalg.norm(a2)
    cos_value = np.round(dot / norm, 15)
    if rad:
        return np.arccos(cos_value)
    else:
        return np.rad2deg(np.arccos(cos_value))


# pnt1 = Pnt([2,3])
# pnt2 = Pnt([2,3,3])
# pnt3 = Pnt([2,3,5])
# pnt31 = Pnt([2,3,5])
# print(pnt31 is pnt3)
# seg1 = Segment(pnt1, pnt2)
# seg11 = Segment(pnt1, pnt2)
# print(seg11 is seg1)
# seg2 = Segment(pnt2, pnt3)
# seg3 = Segment(pnt3, pnt1)
# wire1 = Wire(seg1, seg2,seg3)
# surf1 = Surface(wire1)
# pprint(id_index)
# print(seg3)
# print(pnt1.property["occ_pnt"])


def bend(
    point_cordinates,
    radius: float = None,
    mx_pt: np.ndarray = None,
    mn_pt: np.ndarray = None,
):
    coord_t = np.array(point_cordinates).T
    if mx_pt is None:
        mx_pt = np.max(coord_t, 1)
    if mn_pt is None:
        mn_pt = np.min(coord_t, 1)
    cnt = 0.5 * (mn_pt + mx_pt)
    scale = np.abs(mn_pt - mx_pt)
    if radius is None:
        radius = scale[1] * 2
    o_y = scale[1] * 0.5 + radius
    for pt in point_cordinates:
        xp = pt[0]
        yp = pt[1]
        ratio_l = xp / scale[0]
        ypr = scale[1] * 0.5 - yp
        Rp = radius + ypr
        ly = scale[0] * (1 + ypr / radius)
        lp = ratio_l * ly
        thetp = lp / (Rp)
        thetp = lp / (Rp)
        pt[0] = Rp * np.sin(thetp)
        pt[1] = o_y - Rp * np.cos(thetp)
    return point_cordinates


def project_array(array: np.ndarray, direct: np.ndarray) -> np.ndarray:
    """
    Project an array to the specified direction.
    """
    direct = direct / np.linalg.norm(direct)
    return np.dot(array, direct) * direct


def shortest_distance_point_line(line, p):
    pt1, pt2 = line
    s = pt2 - pt1
    lmbda = (p - pt1).dot(s) / s.dot(s)
    if lmbda < 1 and lmbda > 0:
        pt_compute = pt1 + lmbda * s
        distance = np.linalg.norm(pt_compute - p)
        return lmbda, distance
    elif lmbda <= 0:
        distance = np.linalg.norm(pt1 - p)
        return 0, distance
    else:
        distance = np.linalg.norm(pt2 - p)
        return 1, distance


def bounding_box(pts: list):
    pts = np.array(pts)
    coord_t = np.array(pts).T
    mx_pt = np.max(coord_t, 1)
    mn_pt = np.min(coord_t, 1)
    return mx_pt, mn_pt


def shortest_distance_line_line(
    line1: Union[np.ndarray, Segment], line2: Union[np.ndarray, Segment]
):
    if isinstance(line1, Segment):
        line1 = line1.raw_value
    if isinstance(line2, Segment):
        line2 = line2.raw_value
    pt11, pt12 = line1
    pt21, pt22 = line2
    s1 = pt12 - pt11
    s2 = pt22 - pt21
    s1square = np.dot(s1, s1)
    s2square = np.dot(s2, s2)
    term1 = s1square * s2square - (np.dot(s1, s2) ** 2)
    term2 = s1square * s2square - (np.dot(s1, s2) ** 2)
    if np.isclose(term1, 0) or np.isclose(term2, 0):
        if np.isclose(s1[0], 0):
            s_p = np.array([-s1[1], s1[0], 0])
        else:
            s_p = np.array([s1[1], -s1[0], 0])
        l1 = np.random.randint(1, 4) * 0.1
        l2 = np.random.randint(6, 9) * 0.1
        pt1i = s1 * l1 + pt11
        pt2i = s2 * l2 + pt21
        si = pt2i - pt1i
        dist = np.linalg.norm(
            si * (si * s_p) / (np.linalg.norm(si) * np.linalg.norm(s_p))
        )
        return dist, np.array([pt1i, pt2i])
    lmbda1 = (
        np.dot(s1, s2) * np.dot(pt11 - pt21, s2) - s2square * np.dot(pt11 - pt21, s1)
    ) / (s1square * s2square - (np.dot(s1, s2) ** 2))
    lmbda2 = -(
        np.dot(s1, s2) * np.dot(pt11 - pt21, s1) - s1square * np.dot(pt11 - pt21, s2)
    ) / (s1square * s2square - (np.dot(s1, s2) ** 2))
    condition1 = lmbda1 >= 1
    condition2 = lmbda1 <= 0
    condition3 = lmbda2 >= 1
    condition4 = lmbda2 <= 0
    if condition1 or condition2 or condition3 or condition4:
        choices = [
            [line2, pt11, s2],
            [line2, pt12, s2],
            [line1, pt21, s1],
            [line1, pt22, s1],
        ]
        result = np.zeros((4, 2))
        for i in range(4):
            result[i] = shortest_distance_point_line(choices[i][0], choices[i][1])
        shortest_index = np.argmin(result.T[1])
        shortest_result = result[shortest_index]
        pti1 = (
            shortest_result[0] * choices[shortest_index][2]
            + choices[shortest_index][0][0]
        )
        pti2 = choices[shortest_index][1]
        # print(result)
    else:
        pti1 = pt11 + lmbda1 * s1
        pti2 = pt21 + lmbda2 * s2
    # print(lmbda1, lmbda2)
    # print(pti1, pti2)
    # print(np.dot(s1,pti2 - pti1), np.dot(s2,pti2 - pti1))
    distance = np.linalg.norm(pti1 - pti2)
    return distance, np.array([pti1, pti2])


def check_parallel_line_line(
    line1: Union[np.ndarray, Segment], line2: Union[np.ndarray, Segment]
) -> tuple:
    parallel = False
    colinear = False
    if isinstance(line1, Segment):
        line1 = line1.raw_value
    if isinstance(line2, Segment):
        line2 = line2.raw_value
    pt1, pt2 = line1
    pt3, pt4 = line2

    def generate_link():
        t1 = np.random.randint(1, 4) * 0.1
        t2 = np.random.randint(5, 9) * 0.1
        pt12 = (1 - t1) * pt1 + t1 * pt2
        pt34 = (1 - t2) * pt3 + t2 * pt4
        norm_L3 = np.linalg.norm(pt34 - pt12)
        while np.isclose(norm_L3, 0):
            t1 = np.random.randint(1, 4) * 0.1
            t2 = np.random.randint(5, 9) * 0.1
            pt12 = (1 - t1) * pt1 + t1 * pt2
            pt34 = (1 - t2) * pt3 + t2 * pt4
            norm_L3 = np.linalg.norm(pt34 - pt12)
        return pt12, pt34

    pt12, pt34 = generate_link()
    L1 = (pt2 - pt1) / np.linalg.norm(pt2 - pt1)
    L2 = (pt4 - pt3) / np.linalg.norm(pt4 - pt3)
    L3 = (pt34 - pt12) / np.linalg.norm(pt34 - pt12)
    if np.isclose(np.linalg.norm(np.cross(L1, L2)), 0):
        parallel = True
    if np.isclose(np.linalg.norm(np.cross(L1, L3)), 0) and parallel:
        colinear = True
    return parallel, colinear


def check_overlap(
    line1: Union[np.ndarray, Segment], line2: Union[np.ndarray, Segment]
) -> np.ndarray:
    if isinstance(line1, Segment):
        line1 = line1.raw_value
    if isinstance(line2, Segment):
        line2 = line2.raw_value
    A, B = line1
    C, D = line2
    s = B - A
    dist_s = np.linalg.norm(s)
    norm_s = s / dist_s
    c = C - A
    dist_c = np.linalg.norm(c)
    if np.isclose(dist_c, 0):
        lmbda_c = 0
    else:
        norm_c = c / dist_c
        sign_c = -1 if np.isclose(np.sum(norm_c + norm_s), 0) else 1
        lmbda_c = sign_c * dist_c / dist_s
    d = D - A
    dist_d = np.linalg.norm(d)
    if np.isclose(dist_d, 0):
        lmbda_d = 0
    else:
        norm_d = d / dist_d
        sign_d = -1 if np.isclose(np.sum(norm_d + norm_s), 0) else 1
        lmbda_d = sign_d * dist_d / dist_s
    indicator = np.zeros(4)
    direction_cd = lmbda_d - lmbda_c
    smaller = min(lmbda_c, lmbda_d)
    larger = max(lmbda_c, lmbda_d)
    pnt_list = np.array([A, B, C, D])
    if lmbda_c < 1 and lmbda_c > 0:
        indicator[2] = 1
    if lmbda_d < 1 and lmbda_d > 0:
        indicator[3] = 1
    if 0 < larger and 0 > smaller:
        indicator[0] = 1
    if 1 < larger and 1 > smaller:
        indicator[1] = 1
    return np.where(indicator == 1)[0], np.unique(
        pnt_list[np.where(indicator == 1)[0]], axis=0
    )


def get_face_area(points: list):
    pts = np.array(points).T
    x = pts[0]
    y = pts[1]
    result = 0
    for i in range(len(x)):
        if i < len(x) - 1:
            t = x[i] * y[i + 1] - x[i + 1] * y[i]
        else:
            t = x[i] * y[0] - x[0] * y[i]
        result += t
    return np.abs(result) * 0.5


def get_literal_vector(a: np.ndarray, d: bool):
    """
    @brief This is used to create a vector which is perpendicular to the based vector on its left side ( d = True ) or right side ( d = False )
    @param a vector ( a )
    @param d True if on left or False if on right
    @return A vector.
    """
    z = np.array([0, 0, 1])
    # cross product of z and a
    if d:
        na = np.cross(z, a)
    else:
        na = np.cross(-z, a)
    norm = np.linalg.norm(na, na.shape[0])
    return na / norm


def bisect_angle(
    a1: Union[np.ndarray, Segment], a2: Union[np.ndarray, Segment]
) -> np.ndarray:
    """
    @brief Angular bisector between two vectors. The result is a vector splitting the angle between two vectors uniformly.
    @param a1 1xN numpy array
    @param a2 1xN numpy array
    @return the bisector vector
    """
    if isinstance(a1, Segment):
        a1 = a1.vector
    if isinstance(a2, Segment):
        a2 = a2.vector
    norm1 = np.linalg.norm(a1)
    norm2 = np.linalg.norm(a2)
    bst = a1 / norm1 + a2 / norm2
    norm3 = np.linalg.norm(bst)
    # The laterality indicator a2 norm3 norm3
    if norm3 == 0:
        opt = get_literal_vector(a2, True)
    else:
        opt = bst / norm3
    return opt


def translate(pts: np.ndarray, direct: np.ndarray) -> np.ndarray:
    pts = np.array(
        [
            np.array(list(i.Coord())) if isinstance(i, gp_Pnt) else np.array(i)
            for i in pts
        ]
    )
    pts = [i + direct for i in pts]
    return list(pts)


def center_of_mass(pts: np.ndarray) -> np.ndarray:
    pts = np.array(
        [
            np.array(list(i.Coord())) if isinstance(i, gp_Pnt) else np.array(i)
            for i in pts
        ]
    )

    return np.mean(pts.T, axis=1)


def distance(p1: Pnt, p2: Pnt) -> float:
    return np.linalg.norm(p1.value - p2.value)


def rotate(
    pts: np.ndarray,
    angle_x: float = 0,
    angle_y: float = 0,
    angle_z: float = 0,
    cnt: np.ndarray = None,
) -> np.ndarray:
    pts = np.array(
        [
            np.array(list(i.Coord())) if isinstance(i, gp_Pnt) else np.array(i)
            for i in pts
        ]
    )
    com = center_of_mass(pts)
    if cnt is None:
        cnt = np.array([0, 0, 0])
    t_vec = cnt - com
    pts += t_vec
    rot_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle_x), -np.sin(angle_x)],
            [0, np.sin(angle_x), np.cos(angle_x)],
        ]
    )
    rot_y = np.array(
        [
            [np.cos(angle_y), 0, np.sin(angle_y)],
            [0, 1, 0],
            [-np.sin(angle_y), np.cos(angle_y), 0],
        ]
    )
    rot_z = np.array(
        [
            [np.cos(angle_z), -np.sin(angle_z), 0],
            [np.sin(angle_z), np.cos(angle_z), 0],
            [0, 0, 1],
        ]
    )
    R = rot_x @ rot_y @ rot_z
    rt_pts = pts @ R
    r_pts = rt_pts - t_vec
    return r_pts


def get_center_of_mass(pts: np.ndarray) -> np.ndarray:
    pts = np.array(
        [
            np.array(list(i.Coord())) if isinstance(i, gp_Pnt) else np.array(i)
            for i in pts
        ]
    )

    return np.mean(pts.T, axis=1)


def linear_interpolate(pts: np.ndarray, num: int):
    for i, pt in enumerate(pts):
        if i == len(pts) - 1:
            break
        else:
            interpolated_points = np.linspace(pt, pts[i + 1], num=num + 2)[1:-1]
    return interpolated_points


def interpolate_polygon(
    plg: np.ndarray, step_len: float = None, num: int = None, isclose: bool = True
):
    def deter_dum(line: np.ndarray):
        ratio = step_len / np.linalg.norm(line[0] - line[1])
        if ratio > 0.75:
            num = 0
        elif (ratio > 0.4) and (ratio <= 0.75):
            num = 1
        elif (ratio > 0.3) and (ratio <= 0.4):
            num = 2
        elif (ratio > 0.22) and (ratio <= 0.3):
            num = 3
        elif (ratio > 0.19) and (ratio <= 0.22):
            num = 4
        elif (ratio > 0.14) and (ratio <= 0.19):
            num = 5
        elif ratio <= 0.14:
            num = 7
        return num

    new_plg = plg
    pos = 0
    n = 1
    if not isclose:
        n = 2
    for i, pt in enumerate(plg):
        if i == len(plg) - n:
            break
        line = np.array([pt, plg[i + 1]])
        if num is not None:
            p_num = num
        else:
            p_num = deter_dum(line)
        insert_p = linear_interpolate(line, p_num)
        new_plg = np.concatenate((new_plg[: pos + 1], insert_p, new_plg[pos + 1 :]))
        pos += p_num + 1
    return new_plg


def p_rotate(
    pts: np.ndarray,
    angle_x: float = 0,
    angle_y: float = 0,
    angle_z: float = 0,
    cnt: np.ndarray = None,
) -> np.ndarray:
    pts = np.array(
        [
            np.array(list(i.Coord())) if isinstance(i, gp_Pnt) else np.array(i)
            for i in pts
        ]
    )
    com = get_center_of_mass(pts)
    if cnt is None:
        cnt = np.array([0, 0, 0])
    t_vec = cnt - com
    pts += t_vec
    rot_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle_x), -np.sin(angle_x)],
            [0, np.sin(angle_x), np.cos(angle_x)],
        ]
    )
    rot_y = np.array(
        [
            [np.cos(angle_y), 0, np.sin(angle_y)],
            [0, 1, 0],
            [-np.sin(angle_y), np.cos(angle_y), 0],
        ]
    )
    rot_z = np.array(
        [
            [np.cos(angle_z), -np.sin(angle_z), 0],
            [np.sin(angle_z), np.cos(angle_z), 0],
            [0, 0, 1],
        ]
    )
    R = rot_x @ rot_y @ rot_z
    rt_pts = pts @ R
    r_pts = rt_pts - t_vec
    return r_pts


def get_random_pnt(xmin, xmax, ymin, ymax, zmin=0, zmax=0, numpy_array=True):
    random_x = np.random.randint(xmin, xmax)
    random_y = np.random.randint(ymin, ymax)
    if zmin == 0 and zmax == 0:
        random_z = 0
    else:
        random_z = np.random.randint(zmin, zmax)
    if numpy_array:
        result = np.array([random_x, random_y, random_z])
    else:
        result = Pnt([random_x, random_y, random_z])
    return result


def get_random_line(xmin, xmax, ymin, ymax, zmin=0, zmax=0):
    pt1 = get_random_pnt(xmin, xmax, ymin, ymax, zmin, zmax)
    pt2 = get_random_pnt(xmin, xmax, ymin, ymax, zmin, zmax)
    return np.array([pt1, pt2])


def find_intersect_node_on_edge(line1: Segment, line2: Segment) -> tuple:
    """Find possible intersect node on two lines

    :param line1: The first line
    :type line1: Union[np.ndarray, Segment]
    :param line2: The second line
    :type line2: Union[np.ndarray, Segment]
    """
    parallel, colinear = check_parallel_line_line(line1, line2)
    if parallel:
        if colinear:
            index, coords = check_overlap(line1, line2)
            if len(index) < 4:
                for ind in index:
                    if ind in [0, 1]:
                        pnt_candidate = [line1.start_pnt, line1.end_pnt]
                        return (line2, pnt_candidate[ind])
                    else:
                        pnt_candidate = [line2.start_pnt, line2.end_pnt]
                        return (line1, pnt_candidate[ind - 2])

    else:
        distance = shortest_distance_line_line(line1, line2)
        intersect = np.isclose(distance[0], 0)
        if intersect:
            intersect_pnt = Pnt(distance[1][0])
            return ((line1, intersect_pnt), (line2, intersect_pnt))
