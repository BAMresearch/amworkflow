from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop_VolumeProperties, brepgprop_SurfaceProperties
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.TopoDS import TopoDS_Face, TopoDS_Solid, TopoDS_Shape
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib_Add
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods_Face
from OCC.Core.gp import gp_Pnt
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_WIRE, TopAbs_SHELL, TopAbs_FORWARD, TopAbs_SOLID, TopAbs_COMPOUND
from OCCUtils.Topology  import Topo
import numpy as np

def get_face_center_of_mass(face: TopoDS_Face, gp_pnt: bool = False) -> tuple:
    """
     @brief Get the center of mass of a TopoDS_Face. This is useful for determining the center of mass of a face or to get the centre of mass of an object's surface.
     @param face TopoDS_Face to get the center of mass of
     @param gp_pnt If True return an gp_Pnt object otherwise a tuple of coordinates.
    """
    props = GProp_GProps()
    brepgprop_SurfaceProperties(face, props)
    face_surf = props.CentreOfMass()
    # face_surf. Coord if gp_pnt returns the face surface.
    if gp_pnt:
        return face_surf
    else:
        return face_surf.Coord()

def get_volume_center_of_mass(vol: TopoDS_Solid, gp_pnt: bool = False):
    """
     @brief Return the center of mass of a volume. This is an approximation of the centroid of the volume.
     @param vol TopoDS_Solid object representing a volume
     @param gp_pnt If True return an gp_Pnt object otherwise a tuple of coordinates.
     @return Center of mass of a volume.
    """
    props = GProp_GProps()
    brepgprop_VolumeProperties(vol, props)
    cog = props.CentreOfMass()
    # Return the current coordinate of the current coordinate system.
    if gp_pnt:
        return cog
    else:
        return cog.Coord()
    
def get_face_area(face: TopoDS_Face) -> float:
    """
    @brief Get the area of a TopoDS_Face. This is an approximation of the area of the face.
    @param face to get the area of.
    @return The area of the face.
    """
    props = GProp_GProps()
    brepgprop_SurfaceProperties(face, props)
    face_area = props.Mass()
    return face_area

def get_occ_bounding_box(shape: TopoDS_Shape) -> tuple:
    """
     @brief Get bounding box of occupied space of topo shape.
     @param shape TopoDS_Shape to be searched for occupied space
     @return bounding box of occupied space in x y z coordinates
    """
    bbox = Bnd_Box()
    add_bbox = brepbndlib_Add(shape, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    return xmin, ymin, zmin, xmax, ymax, zmax

def get_faces(_shape):
        """
         @brief Get faces of a shape. This is a list of topods_Face objects that correspond to the faces of the shape
         @param _shape shape to get faces of
         @return list of topods_Face objects ( one for each face in the shape ) for each face
        """
        topExp = TopExp_Explorer()
        topExp.Init(_shape, TopAbs_FACE)
        _faces = []

        # Add faces to the faces list
        while topExp.More():
            fc = topods_Face(topExp.Current())
            _faces.append(fc)
            topExp.Next()

        return _faces
    
def point_coord(p: gp_Pnt) -> tuple:
    """
     @brief Returns the coord of a point. This is useful for debugging and to get the coordinates of an object that is a part of a geometry.
     @param p gp_Pnt to get the coord of
     @return tuple of the coordinate of the point ( x y z ) or None if not a point ( in which case the coordinates are None
    """
    return p.Coord()

def topo_explorer(shape: TopoDS_Shape, shape_type: str) -> list:
    """
     @brief TopoDS Explorer for shape_type. This is a wrapper around TopExp_Explorer to allow more flexibility in the explorer
     @param shape TopoDS_Shape to be explored.
     @param shape_type Type of shape e. g. wire face shell solid compound edge
     @return List of TopoDS_Shape that are explored by shape_type. Example : [ TopoDS_Shape ( " face " ) TopoDS_Shape ( " shell "
    """
    result = []
    map_type = {"wire": TopAbs_WIRE,
                "face": TopAbs_FACE,
                "shell": TopAbs_SHELL,
                "solid": TopAbs_SOLID,
                "compound": TopAbs_COMPOUND,
                "edge": TopAbs_EDGE}
    explorer = TopExp_Explorer(shape, map_type[shape_type])
    # This method is called by the explorer.
    while explorer.More():
        result.append(explorer.Current())
        explorer.Next()
    return result

def traverser(item: TopoDS_Shape) -> Topo:
    return Topo(item)

def p_bounding_box(pts: list):
    pts = np.array(pts)
    coord_t = np.array(pts).T
    mx_pt = np.max(coord_t,1)
    mn_pt = np.min(coord_t,1)
    return mx_pt, mn_pt

def shortest_distance_point_line(line, p):
    pt1, pt2 = line
    s = pt2 - pt1
    lmbda = (p - pt1).dot(s) / s.dot(s)
    pt_compute = pt1 + lmbda * s
    if lmbda < 1 and lmbda > 0:
        distance = np.linalg.norm(pt_compute - p)
        return lmbda, distance
    elif lmbda <= 0:
        distance = np.linalg.norm(pt1 - p)
        return 0, distance
    else:
        distance = np.linalg.norm(pt2 - p)
        return 1, distance
    
def shortest_distance_line_line(line1, line2):
    pt11, pt12 = line1
    pt21, pt22 = line2
    s1 = pt12 - pt11
    s2 = pt22 - pt21
    s1square = np.dot(s1, s1)
    s2square = np.dot(s2, s2)
    lmbda1 = (np.dot(s1,s2) * np.dot(pt11 - pt21,s2) - s2square * np.dot(pt11 - pt21, s1)) / (s1square * s2square - (np.dot(s1, s2)**2))
    lmbda2 = -(np.dot(s1,s2) * np.dot(pt11 - pt21,s1) - s1square * np.dot(pt11 - pt21, s2)) / (s1square * s2square - (np.dot(s1, s2)**2))
    condition1 = lmbda1 >= 1
    condition2 = lmbda1 <= 0
    condition3 = lmbda2 >= 1
    condition4 = lmbda2 <= 0
    if condition1 or condition2 or condition3 or condition4:
        choices = [[line2, pt11,s2], [line2, pt12, s2], [line1, pt21, s1], [line1, pt22, s1]]
        result = np.zeros((4,2))
        for i in range(4):   
            result[i] = shortest_distance_point_line(choices[i][0], choices[i][1])
        shortest_index = np.argmin(result.T[1])
        shortest_result = result[shortest_index]
        pti1 = shortest_result[0] * choices[shortest_index][2] + choices[shortest_index][0][0]
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