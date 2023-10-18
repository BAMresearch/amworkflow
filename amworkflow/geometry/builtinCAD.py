import numpy as np
# from pprint import pprint
from OCC.Core.gp import gp_Pnt
from amworkflow.geometry.simple_geometries import create_edge, create_wire, create_face, create_solid
from amworkflow.occ_helpers import sew_face

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
            6: "compound"
        }

class TopoObj():
    def __init__(self) -> None:
        '''
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
        
        '''
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
                if index != len(item_value)-1:
                    own += f"{item_id}({item_type}),"
                else:
                    own += f"{item_id}({item_type})"
        belong = ""
        for item_type, item_value in self.belong.items():
            item_type = TYPE_INDEX[item_type]
            for index, item_id in enumerate(item_value):
                if index != len(item_value)-1:
                    belong += f"{item_id}({item_type}),"
                else:
                    belong += f"{item_id}({item_type})"
        if self.type == 0:
            value = str(self.value)+"(coordinate)"
        else:
            value = str(self.value)+"(IDs)"
        doc = f"\033[1mType\033[0m: {TYPE_INDEX[self.type]}\n\033[1mID\033[0m: {self.id}\n\033[1mValue\033[0m: {value}\n\033[1mOwn\033[0m: {own}\n\033[1mBelong\033[0m: {belong}\n"
        return doc
        
    def check_value_repetition(self, base_value:any, item_value:any) -> bool:
        '''
        Check if a value is close enough to the base value. True if repeated.
        :param base_value: item to be compared with.
        :param item_value: value to be examined.
        '''
        if type(base_value) is list:
            base_value = np.array(base_value)
            item_value = np.array(item_value)
            
        return np.isclose(np.linalg.norm(base_value-item_value), 0)
   
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
            raise Exception("Wrong geometry object type, perhaps a mistake made in development.")
        return valid

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
    
    def new_item(self, item_value: any, item_type) -> tuple:
        '''
        Check if a value already exits in the index
        :param item_value: value to be examined.
        '''
        for _, item in id_index.items():
            if self.check_type_coincide(item["type"],item_type):
                if self.check_value_repetition(item["value"], item_value):
                    return False, item["id"]
        return True, None
    
    def update_basic_property(self):
        """Update basic properties
        """       
        self.property.update({"type": self.type,
                        "id": self.id,
                        "gid": self.gid,
                        "own": self.own,
                        "belong": self.belong,
                        "value": self.value})
        
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
        '''
        Register an item to the index and return its id. Duplicate value will be filtered.
        :param item_value: value to be registered.
        '''
        new, old_id = self.new_item(self.value, self.type)
        global count_id
        if new:
            self.id = count_id
            self.gid = count_gid[self.type]
            count_gid[self.type] += 1
            self.update_basic_property()
            id_index.update({self.id: self.property})
            count_id += 1
            return self.id
        else:
            return old_id
    
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
                item.belong.update({self.type:[self.id]})
        # self.update_basic_property()
        # self.update_id_index()
        
class Pnt(TopoObj):
    def __init__(self, coord: list) -> None:
        super().__init__()
        self.type = 0
        self.coord = self.pnt(coord)
        self.value = self.coord
        self.occ_pnt = gp_Pnt(*self.coord.tolist())
        self.enrich_property({"occ_pnt": self.occ_pnt})
        self.register_item()
        
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

class Segment(TopoObj):
    def __init__(self, pnt1: Pnt, pnt2: Pnt) -> None:
        super().__init__()
        self.start_pnt = pnt1.id
        self.end_pnt = pnt2.id
        self.type = 1
        self.value = [self.start_pnt, self.end_pnt]
        self.occ_edge = create_edge(pnt1.occ_pnt, pnt2.occ_pnt)
        self.enrich_property({"occ_edge": self.occ_edge})
        self.register_item()
        self.update_dependency(pnt1, pnt2)
        
    def add_relation_to(self, item: "Segment") -> None:
        '''
        Add relation to another segment.
        :param item: item to be added.
        :return: None
        '''
        pass
        

class Wire(TopoObj):
    def __init__(self, *segments: Segment) -> None:
        super().__init__()
        self.type = 2
        self.seg_ids = [item.id for item in segments]
        self.occ_wire = create_wire(*[item.occ_edge for item in segments])
        self.update_dependency(*segments)
        self.value = self.seg_ids
        self.enrich_property({"occ_wire": self.occ_wire})
        self.register_item()
        
class Surface(TopoObj):
    def __init__(self, *wires: Wire) -> None:
        super().__init__()
        self.type = 3
        self.wire_ids = [item.id for item in wires]
        self.value = self.wire_ids
        self.occ_face = create_face(wires[0].occ_wire)
        self.update_dependency(*wires)
        self.enrich_property({"occ_face": self.occ_face})
        self.register_item()
        
class Shell(TopoObj):
    def __init__(self, *surfaces: Surface) -> None:
        super().__init__()
        self.type = 4
        self.surf_ids = [item.id for item in surfaces]
        self.value = self.surf_ids
        self.occ_shell = sew_face(*[item.occ_face for item in surfaces])
        self.update_dependency(*surfaces)
        self.enrich_property({"occ_shell": self.occ_shell})
        self.register_item()
        
class Solid(TopoObj):
    def __init__(self, shell: Shell) -> None:
        super().__init__()
        self.type = 5
        self.shell_id = shell.id
        self.value = [self.shell_id]
        self.occ_solid = create_solid(shell.occ_shell)
        self.update_dependency(shell)
        self.enrich_property({"occ_solid": self.occ_solid})
        self.register_item()
        
# pnt1 = Pnt([2,3])
# pnt2 = Pnt([2,3,3])
# pnt3 = Pnt([2,3,5])
# seg1 = Segment(pnt1, pnt2)
# seg2 = Segment(pnt2, pnt3)
# seg3 = Segment(pnt3, pnt1)
# wire1 = Wire(seg1, seg2,seg3)
# surf1 = Surface(wire1)
# pprint(id_index)
# print(seg3)
# print(pnt1.property["occ_pnt"])



        