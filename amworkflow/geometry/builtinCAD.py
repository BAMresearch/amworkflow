import logging
import multiprocessing
import pickle
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

if multiprocessing.get_start_method(True) != "fork":
    multiprocessing.set_start_method("fork")
level = logging.INFO
logging.basicConfig(
    level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("amworkflow.geometry.builtinCAD")
# logger.setLevel(logging.INFO)
logging.info(f"{multiprocessing.current_process().name} starting.")

count_id = 0
count_gid = [0 for i in range(7)]
component_lib = {}
component_container = {}
TYPE_INDEX = {
    0: "point",
    1: "segment",
    2: "wire",
    3: "surface",
    4: "shell",
    5: "solid",
    6: "compound",
}
manager1 = multiprocessing.Manager()
# manager2 = multiprocessing.Manager()
component_lib = manager1.dict(component_lib)
component_container = manager1.dict(component_container)
count_gid = manager1.list(count_gid)
count_id = manager1.Value("i", count_gid)
TYPE_INDEX = manager1.dict(TYPE_INDEX)


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
            exist_obj_id = self.exist_object.id
            logging.info(
                f"{TYPE_INDEX[gtype]} {exist_obj_id} {gvalue} already exists, return the old one."
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
        for _, item in component_lib.items():
            if self.check_type_coincide(item["type"], item_type):
                if self.check_value_repetition(item["value"], item_value):
                    # return False, item[f"{TYPE_INDEX[item_type]}"]
                    return False, component_container[item["id"]]
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

    def __getstate__(self):
        attributes = self.__dict__.copy()
        return attributes

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
                # f"{TYPE_INDEX[self.type]}": self,
                "id": self.id,
                "gid": self.gid,
                "own": self.own,
                "belong": self.belong,
                "value": self.value,
            }
        )

    def update_component_container(self):
        component_container.update({self.id: self})

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

    def update_component_lib(self):
        component_lib[self.id].update(self.property)

    def register_item(self) -> int:
        """
        Register an item to the index and return its id. Duplicate value will be filtered.
        :param item_value: value to be registered.
        """
        # new, old_id = self.new_item(self.value, self.type)
        global count_id
        global count_gid
        # if new:
        if isinstance(count_id, int):
            self.id = count_id
        else:
            self.id = count_id.value
        self.gid = count_gid[self.type]
        count_gid[self.type] += 1
        self.update_basic_property()
        self.update_component_container()
        component_lib.update({self.id: self.property})
        if isinstance(count_id, int):
            count_id += 1
        else:
            count_id.value += 1
        # Logging the registration
        logging.info(f"Register {TYPE_INDEX[self.type]}: {self.id}")
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
        # self.update_component_lib()


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
    def __new__(cls, pnt1: Pnt = Pnt([]), pnt2: Pnt = Pnt([1])):
        checker = DuplicationCheck(1, [pnt1.id, pnt2.id])
        if checker.new:
            instance = super().__new__(cls)
            instance.re_init = False
            return instance
        else:
            return checker.exist_object

    def __init__(self, pnt1: Pnt = Pnt([]), pnt2: Pnt = Pnt([1])) -> None:
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
            component_lib[self.start_pnt]["value"],
            component_lib[self.end_pnt]["value"],
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
            if pnt2 not in component_lib:
                raise Exception(f"Unrecognized point id: {pnt2}.")
            # pnt2 = component_lib[pnt2]["point"]
            pnt2 = component_container[pnt2]
        if type(pnt1) is int:
            if pnt1 not in component_lib:
                raise Exception(f"Unrecognized point id: {pnt1}.")
            # pnt1 = component_lib[pnt1]["point"]
            pnt1 = component_container[pnt1]
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
# pprint(component_lib)
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
                        return (line2.id, pnt_candidate[ind])
                    else:
                        pnt_candidate = [line2.start_pnt, line2.end_pnt]
                        return (line1.id, pnt_candidate[ind - 2])

    else:
        distance = shortest_distance_line_line(line1, line2)
        intersect = np.isclose(distance[0], 0)
        if intersect:
            intersect_pnt = Pnt(distance[1][0])
            return ((line1.id, intersect_pnt.id), (line2.id, intersect_pnt.id))


class CreateWallByPoints:
    def __init__(self, coords: list, th: float, height: float, is_close: bool = True):
        self.coords = coords
        self.height = height
        self.th = th
        self.is_close = is_close
        self.volume = 0
        self.center_pnts = []
        self.side_pnts = []
        self.center_segments = []
        self.side_segments = []
        self.left_pnts = []
        self.right_pnts = []
        self.left_segments = []
        self.right_segments = []
        self.imaginary_pnt = {}
        self.imaginary_segments = {}
        self.digraph_points = {}
        self.init_points(self.coords, {"position": "center", "active": True})
        self.index_components = {}
        self.edges_to_be_modified = {}
        self.create_sides()
        self.find_overlap_node_on_edge_parallel()
        self.modify_edge()

    def init_points(self, points: list, prop: dict = None):
        for i, p in enumerate(points):
            if not isinstance(p, Pnt):
                p = Pnt(p)
            self.enrich_component(p, prop.copy())
            self.center_pnts.append(p)

    def update_index_component(self):
        for i in component_lib.values():
            if "CWBP" not in i[TYPE_INDEX[i["type"]]].property:
                continue
            property_name = TYPE_INDEX[i["type"]]
            if i[property_name].property["CWBP"]["active"]:
                if i[property_name].id not in self.index_components:
                    self.index_components.update({property_name: [i[property_name].id]})
                else:
                    self.index_component[property_name].append(i[property_name].id)

    def enrich_component(self, component: TopoObj, prop: dict = None) -> TopoObj:
        if not isinstance(component, TopoObj):
            raise ValueError("Component must be a TopoObj")
        component.enrich_property({"CWBP": prop})

    def compute_index(self, ind, length):
        return ind % length if ind > 0 else -(-ind % length)

    def compute_support_vector(self, seg1: Segment, seg2: Segment = None):
        lit_vector = get_literal_vector(seg1.vector, True)
        if seg2 is None:
            th = self.th * 0.5
            angle = np.pi / 2
            sup_vector = lit_vector
        else:
            sup_vector = bisect_angle(-seg2.vector, seg1)
            angle = angle_of_two_arrays(lit_vector, sup_vector)
            if angle > np.pi / 2:
                sup_vector *= -1
                angle = np.pi - angle
            th = np.abs(self.th * 0.5 / np.cos(angle))
        return sup_vector, th

    def create_sides(self):
        close = 1 if self.is_close else 0
        for i in range(len(self.center_pnts) + close):
            # The previous index
            index_l = self.compute_index(i - 1, len(self.center_pnts))
            # The current index
            index = self.compute_index(i, len(self.center_pnts))
            # The next index
            index_n = self.compute_index(i + 1, len(self.center_pnts))
            # The segment from the current index to the next index
            seg_current_next = Segment(
                self.center_pnts[index], self.center_pnts[index_n]
            )
            self.enrich_component(
                seg_current_next, {"position": "center", "active": True}
            )
            # The segment from the previous index to the current index
            seg_previous_current = Segment(
                self.center_pnts[index_l], self.center_pnts[index]
            )
            self.enrich_component(
                seg_previous_current, {"position": "center", "active": True}
            )
            if index == 0 and not self.is_close:
                seg1 = seg_current_next
                seg2 = None
            elif index == len(self.center_pnts) - 1 and not self.is_close:
                seg1 = seg_previous_current
                seg2 = None
            else:
                seg1 = seg_current_next
                seg2 = seg_previous_current
            su_vector, su_len = self.compute_support_vector(seg1, seg2)
            lft_pnt = Pnt(su_vector * su_len + self.center_pnts[index].coord)
            rgt_pnt = Pnt(-su_vector * su_len + self.center_pnts[index].coord)
            self.left_pnts.append(lft_pnt)
            self.enrich_component(lft_pnt, {"position": "left", "active": True})
            self.right_pnts.append(rgt_pnt)
            self.enrich_component(rgt_pnt, {"position": "right", "active": True})
        self.right_pnts = self.right_pnts[::-1]
        self.side_pnts = self.left_pnts + self.right_pnts
        for i in range(len(self.side_pnts) - 1):
            seg_side = Segment(self.side_pnts[i], self.side_pnts[i + 1])
            self.enrich_component(seg_side, {"position": "side", "active": True})
            self.side_segments.append(seg_side)
        self.update_index_component()

    def update_digraph(
        self,
        start_node: int,
        end_node: int,
        insert_node: int = None,
        build_new_edge: bool = True,
    ) -> None:
        self.update_index_component()
        if start_node not in self.index_components:
            raise Exception(f"Unrecognized start node: {start_node}.")
        if end_node not in self.index_components:
            raise Exception(f"Unrecognized end node: {end_node}.")
        if (insert_node not in self.index_components) and (insert_node is not None):
            raise Exception(f"Unrecognized inserting node: {insert_node}.")
        if start_node in self.digraph_points:
            # Directly update the end node
            if insert_node is None:
                self.digraph_points[start_node].append(end_node)
            # Insert a new node between the start and end nodes
            # Basically, the inserted node replaces the end node
            # If build new edge is True, a new edge consists of inserted node and the new end node will be built.
            else:
                end_node_list_index = self.digraph_points[start_node].index(end_node)
                self.digraph_points[start_node][end_node_list_index] = insert_node
                if build_new_edge:
                    edge1 = Segment(
                        component_container[start_node],
                        component_container[insert_node],
                    )
                    self.enrich_component(
                        edge1, {"position": "digraph", "active": True}
                    )
                    edge2 = Segment(
                        component_container[insert_node], component_container[end_node]
                    )
                    self.enrich_component(
                        edge2, {"position": "digraph", "active": True}
                    )
                    self.digraph_points.update({insert_node: [end_node]})
        else:
            # If there is no edge for the start node, a new edge will be built.
            if insert_node is None:
                self.digraph_points.update({start_node: [end_node]})
            else:
                raise Exception("No edge found for insertion option.")

    def find_overlap_node_on_edge_parallel(self):
        visited = []
        line_pairs = []
        for line1 in self.side_segments:
            for line2 in self.side_segments:
                if line1.id != line2.id:
                    if [line1.id, line2.id] not in visited or [
                        line2.id,
                        line1.id,
                    ] not in visited:
                        visited.append([line1.id, line2.id])
                        visited.append([line2.id, line1.id])
                        line_pairs.append((line1, line2))
        num_processes = multiprocessing.cpu_count()
        with multiprocessing.Pool(num_processes) as pool:
            results = pool.starmap(find_intersect_node_on_edge, line_pairs)
            for result in results:
                if result is not None:
                    for pair in result:
                        if pair[0] not in self.edges_to_be_modified:
                            self.edges_to_be_modified.update({pair[0]: [pair[1]]})
                        else:
                            self.edges_to_be_modified[pair[0]].append(pair[1])

    def modify_edge(self):
        for edge_id, nodes_id in self.edges_to_be_modified.items():
            # edge = component_lib[edge_id]["segment"]
            edge = component_container[edge_id]
            nodes = [component_container[i] for i in nodes_id]
            edge_0_coords = edge.raw_value[0]
            nodes_coords = [node.value for node in nodes]
            distances = [np.linalg.norm(i - edge_0_coords) for i in nodes_coords]
            order = np.argsort(distances)
            nodes = [nodes[i] for i in order]
            # component_lib[edge_id]["segment"].property["CWBP"]["active"] = False
            component_container[edge_id].property["CWBP"]["active"] = False
            pts_list = [edge.value[0]] + nodes + [edge.value[1]]
            for i, nd in enumerate(pts_list):
                if i == 0:
                    continue
                self.update_digraph(pts_list[i - 1], nd)
                if (i != 1) and (i != len(pts_list) - 1):
                    if (pts_list[i - 1] in self.imaginary_pnt) and nd in (
                        self.imaginary_pnt
                    ):
                        self.imaginary_segments.update({pts_list[i - 1]: nd})


initialization()
n = 5
# pnt_set = [get_random_pnt(0, 10, 0, 50, 0, 0, numpy_array=False) for i in range(n)]
# pnt_set = [bcad.Pnt([0, 0]), bcad.Pnt([2, 0]), bcad.Pnt([2, 2])]
# cwbp = CreateWallByPoints(pnt_set, 0.5, 0, True)
