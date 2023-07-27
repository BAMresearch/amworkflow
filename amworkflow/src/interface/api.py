import amworkflow.src.geometries.simple_geometry as sg
import amworkflow.src.geometries.composite_geometry as cg
import amworkflow.src.geometries.operator as o
import amworkflow.src.geometries.property as p
import amworkflow.src.geometries.mesher as m
import amworkflow.src.utils.writer as utw
import amworkflow.src.utils.reader as utr
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Wire, TopoDS_Shell, TopoDS_Solid, TopoDS_Face, TopoDS_Edge, topods_Compound
from OCC.Core.gp import gp_Pnt, gp_Vec
from OCC.Core.Geom import Geom_TrimmedCurve
import numpy as np
import gmsh
from amworkflow.src.interface.cli.cli_workflow import cli
import amworkflow.src.infrastructure.database.engine.config as CG
import os
import sys
class amWorkflow(object):
    class engine(object):
        @staticmethod
        def amworkflow(arg):
            args = cli()
            newdir = utw.mk_dir(os.path.dirname(os.path.realpath(__file__)), "test_nd")
            CG.DB_DIR = newdir
            from amworkflow.src.core.workflow import BaseWorkflow
            flow = BaseWorkflow(args = args)
            def inner_decorator(func):
                def wrapped(*args, **kwargs):
                    print('before function')
                    flow.geometry_spawn = func
                    print('after function')
                wrapped()
                return wrapped
            return inner_decorator
    class geom(object):
        @staticmethod
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
            return sg.create_box(length,width,height,radius,alpha,shell)
        
        @staticmethod
        def create_cylinder(radius: float, length: float) -> TopoDS_Shape:
            """
            @brief Create a cylinder shape. This is a convenience function for BRepPrimAPI_MakeCylinder
            @param radius Radius of the cylinder in metres
            @param length Length of the cylinder in metres.
            @return Shape of the cylinder ( TopoDS_Shape ) that is created and ready to be added to topology
            """  
            return sg.create_cylinder(radius,length)
        
        @staticmethod
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
            return sg.create_prism(shape, vector, copy)
        
        @staticmethod
        def create_wire(*edge) -> TopoDS_Wire:
            """
            @brief Create a wire. Input at least one edge to build a wire. This is a convenience function to call BRepBuilderAPI_MakeWire with the given edge and return a wire.
            @return A wire built from the given edge ( s ). The wire may be used in two ways : 1
            """
            return sg.create_wire(edge)
        
        @staticmethod
        def create_edge(pnt1: gp_Pnt = None, pnt2: gp_Pnt = None, arch: Geom_TrimmedCurve = None) -> TopoDS_Edge:
            """
            @brief Create an edge between two points. This is a convenience function to be used in conjunction with : func : ` BRepBuilderAPI_MakeEdge `
            @param pnt1 first point of the edge
            @param pnt2 second point of the edge
            @param arch arch edge ( can be None ). If arch is None it will be created from pnt1 and pnt2
            @return an edge.
            """
            return sg.create_edge(pnt1, pnt2, arch)
        
        @staticmethod
        def create_arch(pnt1, pnt2, pnt1_2, make_edge: bool = True) -> TopoDS_Edge:
            """
            @brief Create an arc of circle. If make_edge is True the arc is created in TopoDS_Edge.
            @param pnt1 The first point of the arc.
            @param pnt2 The second point of the arc.
            @param pnt1_2 The intermediate point of the arc.
            @param make_edge If True the arc is created in the x - y plane.
            @return arch : return an ` GC_MakeArcOfCircle` object or an edge
            """
            return sg.create_arch(pnt1, pnt2, pnt1_2, make_edge)
        
        @staticmethod
        def create_wire_by_points(points: list):
            """
            @brief Create a closed wire (loop) by points. The wire is defined by a list of points which are connected by an edge.
            @param points A list of points. Each point is a gp_Pnt ( x y z) where x, y and z are the coordinates of a point.
            @return A wire with the given points connected by an edge. This will be an instance of : class : `BRepBuilderAPI_MakeWire`
            """
            return sg.create_wire_by_points(points)
        
        @staticmethod
        def random_polygon_constructor(points:list, isface: bool = True) -> TopoDS_Face or TopoDS_Wire:
            """
            @brief Creates a polygon in any shape. If isface is True the polygon is made face - oriented otherwise it is wires
            @param points List of points defining the polygon
            @param isface True if you want to create a face - oriented
            @return A polygon 
            """
            return sg.random_polygon_constructor(points, isface)
        
        @staticmethod
        def angle_of_two_arrays(a1:np.ndarray, a2:np.ndarray, rad: bool = True) -> float:
            """
            @brief Returns the angle between two vectors. This is useful for calculating the rotation angle between a vector and another vector
            @param a1 1D array of shape ( n_features )
            @param a2 2D array of shape ( n_features )
            @param rad If True the angle is in radians otherwise in degrees
            @return Angle between a1 and a2 in degrees or radians depending on rad = True or False
            """
            return sg.angle_of_two_arrays(a1, a2, rad)
        
        @staticmethod
        def create_lateral_vector(a: np.ndarray, d:bool):
            """
            @brief Compute lateral vector of a vector. This is used to create a vector which is perpendicular to the based vector on its left side ( d = True ) or right side ( d = False )
            @param a vector ( a )
            @param d True if on left or False if on right
            @return A vector.
            """
            return sg.laterality_indicator(a, d)
        
        @staticmethod
        def angular_bisector(a1:np.ndarray, a2:np.ndarray) -> np.ndarray:
            """
            @brief Angular bisector between two vectors. The result is a vector splitting the angle between two vectors uniformly.
            @param a1 1xN numpy array
            @param a2 1xN numpy array
            @return the bisector vector
            """
            return sg.angle_of_two_arrays(a1, a2)

        @staticmethod
        def regular_polygon_maker(side_num: int,
                    side_len: float,
                    rotate: float = None,
                    bound: bool = False) -> TopoDS_Face or TopoDS_Wire:
            """
            @brief Creates a regular polygon. The polygon is oriented counterclockwise around the origin. If bound is True the polygon will be only the boundary (TopoDS_Wire) the polygon.
            @param side_num Number of sides of the polygon.
            @param side_len Length of the side of the polygon.
            @param rotate Rotation angle ( in radians ). Defaults to None which means no rotation.
            @param bound output only the boundary. Defaults to False. See documentation for create_wire for more information.
            @return face or boundary of the polygon.
            """
            return cg.polygon_maker(side_num, side_len, rotate, bound)
        
        @staticmethod
        def hexagon_multiplier(side_num: int, side_len: float, iter_num: int, wall: float, center: gp_Pnt = None) -> TopoDS_Face:
            """
            @brief Creates a hexagon with multiplier. This is an iterative approach to the topological sorting algorithm.
            @param side_num Number of sides in the hexagon.
            @param side_len Length of the side ( s ) to be used for the multiplication.
            @param iter_num Number of iterations to perform. Default is 1.
            @param wall Wall thickness.
            @param center Center of the multiplication. Default is original point.
            @return TopoDS_Face. Note that it is the caller's responsibility to check if there is enough space
            """
            return cg.hexagon_multiplier(side_num, side_len, iter_num, wall, center)
        
        @staticmethod
        def isoceles_triangle_maker(bbox_len:float, bbox_wid: float, thickness: float = None) -> TopoDS_Face:
            """
            @brief (Having problem with wall thickness now.) Create isoceles triangulation. This is a function to create isoceles triangulation of a bounding box and its widest corner
            @param bbox_len length of bounding box of the triangle
            @param bbox_wid width of bounding box of the triangle
            @param thickness thickness of the wall of the triangle
            @return a hollowed triangle face.
            """
            return cg.isoceles_triangle_maker(bbox_len, bbox_wid, thickness, float)
        
        @staticmethod
        def create_sym_hexagon1_infill(total_len: float, total_wid:float, height:float, th: float) :
            """
            @brief Create an infill pattern using symmetrical hexagon with defined len, height and numbers.
            @param total_len total length of the bounding box.
            @param total_wid total wid of the bounding box.
            @param height height of the prism. This is the same as height of the hexagon.
            @param th thickness of the wall of the hexagon.
            @return 
            """
            return cg.create_sym_hexagon1_infill(total_len, total_wid, height, th)
        
        @staticmethod
        def create_wall_by_points(pts:list, th: float, isclose:bool, height: float = None, debug: bool = False, output: str = "prism") -> np.ndarray or TopoDS_Face or TopoDS_Shell:
            """
            @brief Create a prism wall by points. It takes a list of points as a skeleton of a central path and then build a strip or a loop.
            @param pts list of 2D points that define the wall. The algorithm can compute points in 3D theoretically but the result may make no sense.
            @param th thickness of the wall.
            @param isclose True if the wall is closed (loop)
            @param height height of the wall if a prism is needed.
            @param debug if True output two groups of points for plotting.
            @param output selecting result intended to output. can be varied among "face" and "prism".
            @return two arrays or a face or a prism.
            """
            return cg.create_wall_by_points(pts, th, isclose, height,debug, output)
        
        @staticmethod
        def get_face_center_of_mass(face: TopoDS_Face, gp_pnt: bool = False) -> tuple:
            """
            @brief Get the center of mass of a TopoDS_Face. This is useful for determining the center of mass of a face or to get the centre of mass of an object's surface.
            @param face TopoDS_Face to get the center of mass of
            @param gp_pnt If True return an gp_Pnt object otherwise a tuple of coordinates.
            """
            return p.get_face_center_of_mass(face, gp_Pnt)
        
        @staticmethod
        def get_face_area(face: TopoDS_Face) -> float:
            """
            @brief Get the area of a TopoDS_Face. This is an approximation of the area of the face.
            @param face to get the area of.
            @return The area of the face.
            """
            return p.get_face_area(face)
        
        @staticmethod
        def get_occ_bounding_box(shape: TopoDS_Shape) -> tuple:
            """
            @brief Get bounding box of occupied space of topo shape.
            @param shape TopoDS_Shape to be searched for occupied space
            @return bounding box of occupied space in x y z coordinates
            """
            return p.get_occ_bounding_box(shape)
        
        @staticmethod
        def get_faces(_shape):
            """
            @brief (This function now can be replaced by topo_explorer(shape, "face").)Get faces from a shape.
            @param _shape shape to get faces of
            @return list of topods_Face objects ( one for each face in the shape ) for each face
            """
            return p.get_faces(_shape)
        
        @staticmethod
        def get_point_coord(_p: gp_Pnt) -> tuple:
            """
            @brief Returns the coord of a point. This is useful for debugging and to get the coordinates of an object that is a part of a geometry.
            @param p gp_Pnt to get the coord of
            @return tuple of the coordinate of the point ( x y z ) or None if not a point ( in which case the coordinates are None
            """
            return p.point_coord(_p)
        
        @staticmethod
        def topo_explorer(shape: TopoDS_Shape, shape_type: str) -> list:
            """
            @brief TopoDS Explorer for shape_type. This is a wrapper around TopExp_Explorer to allow more flexibility in the explorer
            @param shape TopoDS_Shape to be explored.
            @param shape_type Type of shape e. g. wire face shell solid compound edge
            @return List of TopoDS_Shape that are explored by shape_type. Example : [ TopoDS_Shape ( " face " ) TopoDS_Shape ( " shell "
            """
            return p.topo_explorer(shape, shape_type)
        
        @staticmethod
        def rotate(shape: TopoDS_Shape, angle: float, axis: str = "z"):
            """
            @brief Rotate the topography by the given angle around the center of mass of the face.
            @param shape TopoDS_Shape to be rotated.
            @param angle Angle ( in degrees ) to rotate by.
            @param axis determine the rotation axis.
            @return the rotated shape.
            """
            return o.rotate_face(shape, angle, axis)
        
        @staticmethod
        def fuse(shape1: TopoDS_Shape, shape2: TopoDS_Shape) -> TopoDS_Shape:
            """
            @brief Fuse two shapes into one.
            @param shape1 first shape to fuse.
            @param shape2 second shape to fuse.
            @return topoDS_Shape
            """
            return o.fuser(shape1, shape2)
        
        @staticmethod
        def cutter3D(shape1: TopoDS_Shape, shape2: TopoDS_Shape) -> TopoDS_Shape:
            """
            @brief Cut a TopoDS_Shape from shape1 by shape2. It is possible to use this function to cut an object in 3D
            @param shape1 shape that is to be cut
            @param shape2 shape that is to be cut. It is possible to use this function to cut an object in 3D
            @return a shape that is the result of cutting shape1 by shape2 ( or None if shape1 and shape2 are equal
            """
            return o.cutter3D(shape1, shape2)
        
        @staticmethod
        def common(shape1: TopoDS_Shape, shape2: TopoDS_Shape) -> TopoDS_Shape:
            """
            @brief Common between two TopoDS_Shapes. The result is a shape that has all components of shape1 and shape2
            @param shape1 the first shape to be compared
            @param shape2 the second shape to be compared ( must be same shape! )
            @return the common shape or None if there is no common shape between shape1 and shape2 in the sense that both shapes are
            """
            return o.common(shape1, shape2)
        
        @staticmethod
        def translate(item: TopoDS_Shape,
                vector: list) -> None:
            """
            @brief Translates the shape by the distance and direction of a given vector.
            @param item The item to be translated. It must be a TopoDS_Shape
            @param vector The vector to translate the object by. The vector has to be a list with three elements
            """
            return o.translate(item, vector)
        
        @staticmethod
        def reverse(item:TopoDS_Shape) -> None:
            """
            @brief Reverse the shape.
            @param item The item to reverse.
            @return The reversed item
            """
            o.reverse(item)
            
        @staticmethod
        def geom_copy(item: TopoDS_Shape):
            """
            @brief Copy a geometry to a new shape. This is a wrapper around BRepBuilderAPI_Copy and can be used to create a copy of a geometry without having to re - create the geometry in the same way.
            @param item Geometry to be copied.
            @return New geometry that is a copy of the input geometry.
            """
            return o.geom_copy(item)
        
        @staticmethod
        def split(item: TopoDS_Shape, 
          nz: int = None, 
          layer_thickness: float = None,
          split_z: bool = True, 
          split_x: bool = False, 
          split_y: bool = False, 
          nx: int = None, 
          ny: int = None):
            """
            @brief Split a TopoDS_Shape into sub - shapes. 
            @param item TopoDS_Shape to be split.
            @param nz Number of z - points to split.
            @param layer_thickness Layer thickness ( m ).
            @param split_z Split on the Z direction.
            @param split_x Split on the X direction.
            @param split_y Split on the Y direction.
            @param nx Number of sub - shapes in the x - direction.
            @param ny Number of sub - shapes in the y - direction.
            @return a compound of sub-shapes
            """
            return o.split(item, layer_thickness, split_x, split_y, split_z, nx, ny)
        
        @staticmethod
        def intersector(item: TopoDS_Shape,
                     position: float,
                     axis: str) -> TopoDS_Shape:
            """
            @brief Returns the topo shape intersecting the item at the given position.
            @param position Position of the plane in world coordinates.
            @param axis Axis along which of the direction.
            @return TopoDS_Shape with intersection or empty TopoDS_Shape if no intersection is found.
            """
            return o.intersector(item, position, axis)
        
        @staticmethod
        def scale(item: TopoDS_Shape, cnt_pnt: gp_Pnt, factor: float) -> TopoDS_Shape:
            """
            @brief Scales TopoDS_Shape to a given value. This is useful for scaling shapes that are in a shape with respect to another shape.
            @param item TopoDS_Shape to be scaled.
            @param cnt_pnt the point of the scaling center.
            @param factor Factor to scale the shape by. Default is 1.
            @return a scaled TopoDS_Shape with scaling applied to it.
            """
            return o.scaler(item, cnt_pnt, factor)
        
        @staticmethod
        def hollow_carve(face: TopoDS_Shape, factor: float):
            """
            @brief (This can be replaced by cutter3D() now.)Carving on a face with a shape scaling down from itself.
            @param face TopoDS_Shape to be cut.
            @param factor Factor to be used to scale the cutter.
            @return A shape with the cutter in it's center of mass scaled by factor
            """
            return o.hollow_carver(face, factor)
        
    class mesh(object):
        @staticmethod
        def gmsh_switch(s: bool) -> None:
            """
            @brief
            """
            return m.gmsh_switch(s)
        
        @staticmethod
        def get_geom_pointer(model: gmsh.model, shape: TopoDS_Shape) -> list:
            return m.get_geom_pointer(model, shape)
            
        @staticmethod
        def mesher(item: TopoDS_Shape,
           model_name: str,
           layer_type: bool,
           layer_param : float = None,
           size_factor: float = 0.1) -> gmsh.model :
            return m.mesher(item, model_name, layer_type, layer_param, size_factor)
        
    class tool(object):
        @staticmethod
        def stl_writer(item: any, item_name: str, linear_deflection: float = 0.001, angular_deflection: float = 0.1, output_mode = 1, store_dir: str = None) -> None:
            """
            @brief Write OCC to STL file. This function is used to write a file to the database. The file is written to a file named item_name. 
            @param item the item to be written to the file.
            @param item_name the name of the item. It is used to generate the file name.
            @param linear_deflection the linear deflection factor.
            @param angular_deflection the angular deflection factor.
            @param output_mode for using different api in occ.
            @param store_dir the directory to store the file in.
            @return None if success else error code ( 1 is returned if error ). In case of error it is possible to raise an exception
            """
            utw.stl_writer(item, item_name, linear_deflection, angular_deflection, output_mode, store_dir)

        @staticmethod
        def step_writer(item: any, filename: str):
            """
            @brief Writes a step file. This is a wrapper around write_step_file to allow a user to specify the shape of the step and a filename
            @param item the item to write to the file
            @param filename the filename to write the file to ( default is None
            """
            utw.step_writer(item, filename)
        
        @staticmethod
        def namer(name_type: str,
          dim_vector: np.ndarray = None,
          batch_num: int = None,
          parm_title: list = None,
          is_layer_thickness: bool = None,
          layer_param: float or int = None,
          geom_name: str = None
          ) -> str:
            """
            @brief Generate a name based on the type of name. It is used to generate an output name for a layer or a geometric object
            @param name_type Type of name to generate
            @param dim_vector Vector of dimension values ( default : None )
            @param batch_num Number of batch to generate ( default : None )
            @param parm_title List of parameters for the layer
            @param is_layer_thickness True if the layer is thickness ( default : False )
            @param layer_param Parameter of the layer ( default : None )
            @param geom_name Name of the geometric object ( default : None )
            @return Name of the layer or geometric object ( default : None ) - The string representation of the nam
            """
            return utw.namer(name_type, dim_vector, batch_num, parm_title, is_layer_thickness, layer_param, geom_name)
        
        @staticmethod
        def get_filename(path: str) -> str:
            return utr.get_filename(path)
        
        @staticmethod
        def step_reader(path: str) -> TopoDS_Shape():
            return utr.step_reader(path)
        
        @staticmethod
        def stl_reader(path: str) -> TopoDS_Shape:
            return utr.stl_reader(path)
        
        
    
    