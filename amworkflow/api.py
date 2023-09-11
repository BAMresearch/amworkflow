import amworkflow.src.geometries.simple_geometry as sg
import amworkflow.src.geometries.composite_geometry as cg
import amworkflow.src.geometries.operator as o
import amworkflow.src.geometries.property as p
import amworkflow.src.geometries.mesher as m
import amworkflow.src.geometries.builder as b
import amworkflow.src.utils.writer as utw
import amworkflow.src.utils.reader as utr
import amworkflow.src.infrastructure.database.cruds.crud as cr
import amworkflow.src.utils.db_io as dio
import amworkflow.src.interface.gui.Qt.draft_ui as dui
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Wire, TopoDS_Shell, TopoDS_Solid, TopoDS_Face, TopoDS_Edge, TopoDS_Compound
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Pln, gp_Dir
from OCC.Core.Geom import Geom_TrimmedCurve
from OCCUtils.Topology  import Topo
from OCCUtils.Construct import vec_to_dir
import numpy as np
import gmsh
from amworkflow.src.interface.cli.cli_workflow import cli
from amworkflow.src.constants.data_model import DeepMapParamModel
import amworkflow.src.infrastructure.database.engine.config as cfg
import os
import sys
import inspect
import pandas as pd
import importlib


class amWorkflow(object):
    class engine(object):
        @staticmethod
        def amworkflow(mode: str = "production"):
            '''
            Engine to run the entire workflow.
            '''
            args = DeepMapParamModel(cli().__dict__)
            args.mode = mode
            caller_frame = inspect.stack()[1]
            caller_fullpath = caller_frame.filename
            dbdir = utw.mk_dir(os.path.dirname(caller_fullpath), "db")
            args.db_dir = dbdir
            args.import_file_dir = utw.mk_dir(dbdir, "imports")
            args.db_opt_dir = utw.mk_dir(os.path.dirname(caller_fullpath), "output")
            args.db_file_dir = utw.mk_dir(dbdir, "files")
            cfg.DB_DIR = dbdir
            from amworkflow.src.core.workflow import BaseWorkflow
            flow = BaseWorkflow(args = args)
            def inner_decorator(func):
                def wrapped(*args, **kwargs):
                    flow.geometry_spawn = func
                    i = flow.indicator
                    if i[0] == 3:
                        dui.draft_ui(func,flow)
                    else:
                        if (i[4] == 1) or flow.onlyimport:
                            flow.create()
                        if flow.cmesh:
                            flow.mesh()
                        if flow.pcs_indicator[0] == 1:
                            flow.auto_download()
                wrapped()
                return wrapped
            return inner_decorator
    class db(object):
        @staticmethod
        def query_data(table: str, by_name: str = None, column_name: str = None, snd_by_name: str = None, snd_column_name: str = None, only_for_column: str = None) -> pd.DataFrame:
            return cr.query_multi_data(table=table, by_name=by_name, column_name=column_name, target_column_name=only_for_column, snd_by_name=snd_by_name, snd_column_name=snd_column_name)
        
        @staticmethod
        def query_data_obj(table: str, by_name: str, column_name: str):
            return cr.query_data_object(table=table, by_name=by_name,column_name=column_name)
        
        @staticmethod
        def delete_data(table: str, prim_ky: str | list = None, by_name: str = None, column_name: str = None, isbatch: bool = False) -> None:
            return cr.delete_data(table=table, by_primary_key=prim_ky, by_name=by_name, column_name=column_name, isbatch=isbatch)
        
        @staticmethod
        def update_data(table: str, by_name: str | list, on_column: str, edit_column: str, new_value: int | str | float | bool, isbatch: bool = False):
            return cr.update_data(table=table, by_name=by_name, target_column=on_column, new_value=new_value, isbatch=isbatch, edit_column=edit_column)
        
        @staticmethod
        def insert_data(table: str, data: dict, isbatch: bool = False) -> None:
            return cr.insert_data(table=table, data=data, isbatch=isbatch)
        
        @staticmethod
        def have_data_in_db(table: str, column_name, dataset: list, filter_by: str = None, search_column: str = None, filter_by2: str = None, search_column2: str = None) -> bool | list:
            return utr.having_data(table=table, column_name=column_name,dataset=dataset, filter=filter_by, search_column=search_column, filter2=filter_by2, search_column2=search_column2)
        
        @staticmethod
        def query_join_data(table: str, join_column: str, table1: str,  join_column1:str, table2: str = None, join_column2:str = None, filter0: str = None, filter1: str = None, filter2: str = None, on_column_tb: str = None, on_column_tb1: str = None, on_column_tb2: str = None):
            return cr.query_join_tables(table=table, join_column=join_column, table1=table1, join_column1=join_column1, table2=table2, filter0=filter0, filter1=filter1, filter2=filter2, on_column_tb=on_column_tb, on_column_tb1=on_column_tb1, on_column_tb2=on_column_tb2, join_column2=join_column2)
        
    class geom(object):
        @staticmethod
        def make_compound(*args) -> TopoDS_Compound:
            return b.geometry_builder(*args)
        
        @staticmethod
        def sew(*component) -> TopoDS_Shape:
            return b.sewer(*component)
        
        @staticmethod
        def make_solid(item: TopoDS_Shape) -> TopoDS_Shape:
            return b.solid_maker(item=item)
        @staticmethod
        def pnt(x: float, y:float, z:float = 0) -> gp_Pnt:
            return gp_Pnt(x, y, z)
        
        @staticmethod
        def vec(x: float, y:float, z:float = 0) -> gp_Vec:
            return gp_Vec(x,y,z)
        
        @staticmethod
        def plane(pnt: gp_Pnt, vec: gp_Vec) -> gp_Pln:
            return gp_Pln(pnt, vec)
        
        @staticmethod
        def vec2dir(vec: gp_Vec) -> gp_Dir:
            return vec_to_dir(vec)
        
        @staticmethod
        def create_face(wire: TopoDS_Wire) -> TopoDS_Face:
            return sg.create_face(wire=wire)
        
        @staticmethod
        def create_box(length: float, 
               width: float, 
               height: float, 
               radius: float = None,
               alpha: float = None,
               shell: bool = False) -> TopoDS_Shape:
            """
            Create a box with given length width height and radius. If radius is None or 0 the box will be sewed by a solid.
            :param length Length of the box in points
            :param width Width of the box.
            :param height Height of the box.
            :param radius Radius of the box. Default is None which means that the box is without curves.
            :param alpha defines the angle of bending the box. Default is half the length divided by the radius.
            :param shell If True the box will be shell. Default is False.
            :return: TopoDS_Shape with box in it's topolar form. Note that this is a Shape
            """
            return sg.create_box(length,width,height,radius,alpha,shell)
        
        @staticmethod
        def create_cylinder(radius: float, length: float) -> TopoDS_Shape:
            """
            Create a cylinder shape. This is a convenience function for BRepPrimAPI_MakeCylinder
            :param radius Radius of the cylinder in metres
            :param length Length of the cylinder in metres.
            :return: Shape of the cylinder ( TopoDS_Shape ) that is created and ready to be added to topology
            """  
            return sg.create_cylinder(radius,length)
        
        @staticmethod
        def create_prism(shape: TopoDS_Shape,
                    vector: list,
                    copy: bool = True) -> TopoDS_Shell:
            """
            Create prism from TopoDS_Shape and vector. It is possible to copy the based wire(s) if copy is True. I don't know what if it's False so it is recommended to always use True.
            :param shape TopoDS_Shape to be used as base
            :param vector list of 3 elements ( x y z ). Normally only use z to define the height of the prism.
            :param copy boolean to indicate if the shape should be copied
            :return: return the prism
            """
            return sg.create_prism(shape, vector, copy)
        
        @staticmethod
        def create_prism_by_curve(shape: TopoDS_Shape, curve: TopoDS_Wire):
            return sg.create_prism_by_curve(shape, curve)
        
        @staticmethod
        def create_face_by_plane(pln: gp_Pln, *vt: gp_Pnt) -> TopoDS_Face:
            return sg.create_face_by_plane(pln, *vt)
        
        @staticmethod
        def create_wire(*edge) -> TopoDS_Wire:
            """
            Create a wire. Input at least one edge to build a wire. This is a convenience function to call BRepBuilderAPI_MakeWire with the given edge and return a wire.
            :return: A wire built from the given edge ( s ). The wire may be used in two ways : 1
            """
            return sg.create_wire(edge)
        
        @staticmethod
        def create_edge(pnt1: gp_Pnt = None, pnt2: gp_Pnt = None, arch: Geom_TrimmedCurve = None) -> TopoDS_Edge:
            """
            Create an edge between two points. This is a convenience function to be used in conjunction with : func : ` BRepBuilderAPI_MakeEdge `
            :param pnt1 first point of the edge
            :param pnt2 second point of the edge
            :param arch arch edge ( can be None ). If arch is None it will be created from pnt1 and pnt2
            :return: an edge.
            """
            return sg.create_edge(pnt1, pnt2, arch)
        
        @staticmethod
        def create_arch(pnt1, pnt2, pnt1_2, make_edge: bool = True) -> TopoDS_Edge:
            """
            Create an arc of circle. If make_edge is True the arc is created in TopoDS_Edge.
            :param pnt1 The first point of the arc.
            :param pnt2 The second point of the arc.
            :param pnt1_2 The intermediate point of the arc.
            :param make_edge If True the arc is created in the x - y plane.
            :return: arch : return an ` GC_MakeArcOfCircle` object or an edge
            """
            return sg.create_arch(pnt1, pnt2, pnt1_2, make_edge)
        
        @staticmethod
        def create_wire_by_points(points: list):
            """
            Create a closed wire (loop) by points. The wire is defined by a list of points which are connected by an edge.
            :param points A list of points. Each point is a gp_Pnt ( x y z) where x, y and z are the coordinates of a point.
            :return: A wire with the given points connected by an edge. This will be an instance of : class : `BRepBuilderAPI_MakeWire`
            """
            return sg.create_wire_by_points(points)
        
        @staticmethod
        def random_polygon_constructor(points:list, isface: bool = True) -> TopoDS_Face or TopoDS_Wire:
            """
            Creates a polygon in any shape. If isface is True the polygon is made face - oriented otherwise it is wires
            :param points List of points defining the polygon
            :param isface True if you want to create a face - oriented
            :return: A polygon 
            """
            return sg.random_polygon_constructor(points, isface)
        
        @staticmethod
        def angle_of_two_arrays(a1:np.ndarray, a2:np.ndarray, rad: bool = True) -> float:
            """
            Returns the angle between two vectors. This is useful for calculating the rotation angle between a vector and another vector
            :param a1 1D array of shape ( n_features )
            :param a2 2D array of shape ( n_features )
            :param rad If True the angle is in radians otherwise in degrees
            :return: Angle between a1 and a2 in degrees or radians depending on rad = True or False
            """
            return sg.angle_of_two_arrays(a1, a2, rad)
        
        @staticmethod
        def create_lateral_vector(a: np.ndarray, d:bool):
            """
            Compute lateral vector of a vector. This is used to create a vector which is perpendicular to the based vector on its left side ( d = True ) or right side ( d = False )
            :param a vector ( a )
            :param d True if on left or False if on right
            :return: A vector.
            """
            return sg.laterality_indicator(a, d)
        
        @staticmethod
        def angular_bisector(a1:np.ndarray, a2:np.ndarray) -> np.ndarray:
            """
            Angular bisector between two vectors. The result is a vector splitting the angle between two vectors uniformly.
            :param a1 1xN numpy array
            :param a2 1xN numpy array
            :return: the bisector vector
            """
            return sg.angular_bisector(a1, a2)

        @staticmethod
        def make_regular_polygon(side_num: int,
                    side_len: float,
                    rotate: float = None,
                    bound: bool = False) -> TopoDS_Face or TopoDS_Wire:
            """
            Creates a regular polygon. The polygon is oriented counterclockwise around the origin. If bound is True the polygon will be only the boundary (TopoDS_Wire) the polygon.
            :param side_num Number of sides of the polygon.
            :param side_len Length of the side of the polygon.
            :param rotate Rotation angle ( in radians ). Defaults to None which means no rotation.
            :param bound output only the boundary. Defaults to False. See documentation for create_wire for more information.
            :return: face or boundary of the polygon.
            """
            return cg.polygon_maker(side_num, side_len, rotate, bound)
        
        @staticmethod
        def multiply_hexagon(side_num: int, side_len: float, iter_num: int, wall: float, center: gp_Pnt = None) -> TopoDS_Face:
            """
            Creates a hexagon with multiplier. This is an iterative approach to the topological sorting algorithm.
            :param side_num Number of sides in the hexagon.
            :param side_len Length of the side ( s ) to be used for the multiplication.
            :param iter_num Number of iterations to perform. Default is 1.
            :param wall Wall thickness.
            :param center Center of the multiplication. Default is original point.
            :return: TopoDS_Face. Note that it is the caller's responsibility to check if there is enough space
            """
            return cg.hexagon_multiplier(side_num, side_len, iter_num, wall, center)
        
        @staticmethod
        def make_isoceles_triangle(bbox_len:float, bbox_wid: float, thickness: float = None) -> TopoDS_Face:
            """
            (Having problem with wall thickness now.) Create isoceles triangulation. This is a function to create isoceles triangulation of a bounding box and its widest corner
            :param bbox_len length of bounding box of the triangle
            :param bbox_wid width of bounding box of the triangle
            :param thickness thickness of the wall of the triangle
            :return: a hollowed triangle face.
            """
            return cg.isoceles_triangle_maker(bbox_len, bbox_wid, thickness)
        
        @staticmethod
        def create_sym_hexagon1_infill(total_len: float, total_wid:float, height:float, th: float) :
            """
            Create an infill pattern using symmetrical hexagon with defined len, height and numbers.
            :param total_len total length of the bounding box.
            :param total_wid total wid of the bounding box.
            :param height height of the prism. This is the same as height of the hexagon.
            :param th thickness of the wall of the hexagon.
            :return: 
            """
            return cg.create_sym_hexagon1_infill(total_len, total_wid, height, th)
        
        # @staticmethod
        # def create_wall_by_points(pts:list, th: float, isclose:bool, height: float = None, debug: bool = False, debug_type: str = "linear", output: str = "prism", interpolate: float = None, R: float = None) -> np.ndarray or TopoDS_Face or TopoDS_Shell:
        #     """
        #     Create a prism wall by points. It takes a list of points as a skeleton of a central path and then build a strip or a loop.
        #     :param pts: list of 2D points that define the wall. The algorithm can compute points in 3D theoretically but the result may make no sense.
        #     :param th: thickness of the wall.
        #     :param isclose: True if the wall is closed (loop)
        #     :param height: height of the wall if a prism is needed.
        #     :param debug: if True output two groups of points for plotting.
        #     :param output: selecting result intended to output. can be varied among "face" and "prism".
        #     :return: two arrays or a face or a prism.
        #     """
        #     return cg.create_wall_by_points(pts, th, isclose, height,debug, debug_type, output,interpolate, R)
        
        # @staticmethod
        class CreateWallByPoints(cg.CreateWallByPointsUpdate):
            def __init__(self, pts: list, th: float, height: float, is_close:bool = True):
                super().__init__(pts,th,height,is_close)
        
        @staticmethod
        def get_face_center_of_mass(face: TopoDS_Face, gp_pnt: bool = False) -> tuple | gp_Pnt:
            """
            Get the center of mass of a TopoDS_Face. This is useful for determining the center of mass of a face or to get the centre of mass of an object's surface.
            :param face TopoDS_Face to get the center of mass of
            :param gp_pnt If True return an gp_Pnt object otherwise a tuple of coordinates.
            """
            return p.get_face_center_of_mass(face, gp_pnt)
        
        @staticmethod
        def get_face_area(face: TopoDS_Face) -> float:
            """
            Get the area of a TopoDS_Face. This is an approximation of the area of the face.
            :param face: to get the area of.
            :return: The area of the face.
            """
            return p.get_face_area(face)
        
        @staticmethod
        def get_occ_bounding_box(shape: TopoDS_Shape) -> tuple:
            """
            Get bounding box of occupied space of topo shape.
            :param shape: TopoDS_Shape to be searched for occupied space
            :return: bounding box of occupied space in x y z coordinates
            """
            return p.get_occ_bounding_box(shape)
        
        @staticmethod
        def get_faces(_shape):
            """
            (This function now can be replaced by topo_explorer(shape, "face").)Get faces from a shape.
            :param _shape: shape to get faces of
            :return: list of topods_Face objects ( one for each face in the shape ) for each face
            """
            return p.get_faces(_shape)
        
        @staticmethod
        def get_boundary(item: TopoDS_Shape) -> TopoDS_Wire:
            '''
            Get the boundary line of a rectangle shape.
            :param item: Item to be inspected.
            '''
            return o.get_boundary(item=item)
        @staticmethod
        def get_point_coord(_p: gp_Pnt) -> tuple:
            """
            Returns the coord of a point. This is useful for debugging and to get the coordinates of an object that is a part of a geometry.
            :param p: gp_Pnt to get the coord of
            :return: tuple of the coordinate of the point ( x y z ) or None if not a point ( in which case the coordinates are None
            """
            return p.point_coord(_p)
        
        @staticmethod
        def topo_explorer(shape: TopoDS_Shape, shape_type: str) -> list:
            """
            TopoDS Explorer for shape_type. This is a wrapper around TopExp_Explorer to allow more flexibility in the explorer
            :param shape: TopoDS_Shape to be explored.
            :param shape_type: Type of shape e. g. wire face shell solid compound edge
            :return: List of TopoDS_Shape that are explored by shape_type. Example : [ TopoDS_Shape ( " face " ) TopoDS_Shape ( " shell "
            """
            return p.topo_explorer(shape, shape_type)
        
        @staticmethod
        def traverse(item: TopoDS_Shape) -> Topo:
            '''Traverse the whole item and return how the topological parts connecting with each other.
            :param item: item to be traversed.
            :return: An instance of Topo class. To get for example all solids from the instacne, do foo.solids(), which will return an python iterator of all solids. 
            
            tips: If necessary, it can be converted to a list by list(<iterator>)
            
            '''
            return p.traverser(item=item)
        
        @staticmethod
        def rotate(shape: TopoDS_Shape, angle: float, axis: str = "z"):
            """
            :brief: Rotate the topography by the given angle around the center of mass of the face.
            :param shape: TopoDS_Shape to be rotated.
            :param angle: Angle ( in degrees ) to rotate by.
            :param axis: determine the rotation axis.
            :return: the rotated shape.
            """
            return o.rotate_face(shape, angle, axis)
        
        @staticmethod
        def fuse(shape1: TopoDS_Shape, shape2: TopoDS_Shape) -> TopoDS_Shape:
            """
            :brief: Fuse two shapes into one.
            :param shape1: first shape to fuse.
            :param shape2: second shape to fuse.
            :return: topoDS_Shape
            """
            return o.fuser(shape1, shape2)
        
        @staticmethod
        def cutter3D(shape1: TopoDS_Shape, shape2: TopoDS_Shape) -> TopoDS_Shape:
            """
            Cut a TopoDS_Shape from shape1 by shape2. It is possible to use this function to cut an object in 3D
            :param shape1: shape that is to be cut
            :param shape2: shape that is to be cut. It is possible to use this function to cut an object in 3D
            :return: a shape that is the result of cutting shape1 by shape2 ( or None if shape1 and shape2 are equal
            """
            return o.cutter3D(shape1, shape2)
        
        @staticmethod
        def common(shape1: TopoDS_Shape, shape2: TopoDS_Shape) -> TopoDS_Shape:
            """
            Common between two TopoDS_Shapes. The result is a shape that has all components of shape1 and shape2
            :param shape1: the first shape to be compared
            :param shape2: the second shape to be compared ( must be same shape! )
            :return: the common shape or None if there is no common shape between shape1 and shape2 in the sense that both shapes are
            """
            return o.common(shape1, shape2)
        
        @staticmethod
        def translate(item: TopoDS_Shape,
                vector: list) -> None:
            """
            Translates the shape by the distance and direction of a given vector.
            :param item: The item to be translated. It must be a TopoDS_Shape
            :param vector: The vector to translate the object by. The vector has to be a list with three elements
            """
            return o.translate(item, vector)
        
        @staticmethod
        def p_translate(pts:np.ndarray, direct: np.ndarray) -> np.ndarray:
            return sg.p_translate(pts, direct=direct)
        
        @staticmethod
        def p_rotate(pts:np.ndarray, angle_x: float = 0, angle_y: float = 0, angle_z: float = 0, cnt:np.ndarray = None) -> np.ndarray:
            return sg.p_rotate(pts = pts, cnt = cnt, angle_x = angle_x, angle_y=angle_y, angle_z=angle_z)
        
        @staticmethod
        def p_center_of_mass(pts: np.ndarray) -> np.ndarray:
            return sg.p_center_of_mass(pts=pts)
        
        @staticmethod
        def reverse(item:TopoDS_Shape) -> TopoDS_Shape:
            """
            Reverse the shape.
            :param item: The item to reverse.
            :return: The reversed item
            """
            o.reverse(item)
            return item
            
        @staticmethod
        def geom_copy(item: TopoDS_Shape):
            """
            Copy a geometry to a db shape. This is a wrapper around BRepBuilderAPI_Copy and can be used to create a copy of a geometry without having to re - create the geometry in the same way.
            :param item: Geometry to be copied.
            :return: db geometry that is a copy of the input geometry.
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
            Split a TopoDS_Shape into sub - shapes. 
            :param item: TopoDS_Shape to be split.
            :param nz: Number of z - points to split.
            :param layer_thickness: Layer thickness ( m ).
            :param split_z: Split on the Z direction.
            :param split_x: Split on the X direction.
            :param split_y: Split on the Y direction.
            :param nx: Number of sub - shapes in the x - direction.
            :param ny: Number of sub - shapes in the y - direction.
            :return: a compound of sub-shapes
            """
            return o.split(item=item, nz=nz, layer_thickness=layer_thickness, split_x=split_x, split_y=split_y, split_z=split_z, nx=nx, ny=ny)
        
        @staticmethod
        def split2(item: TopoDS_Shape, *tools: TopoDS_Shape) -> TopoDS_Compound:
            return o.split2(item,*tools)
        
        @staticmethod
        def intersector(item: TopoDS_Shape,
                     position: float,
                     axis: str) -> TopoDS_Shape:
            """
            Returns the topo shape intersecting the item at the given position.
            :param position: Position of the plane in world coordinates.
            :param axis: Axis along which of the direction.
            :return: TopoDS_Shape with intersection or empty TopoDS_Shape if no intersection is found.
            """
            return o.intersector(item, position, axis)
        
        @staticmethod
        def scale(item: TopoDS_Shape, cnt_pnt: gp_Pnt, factor: float) -> TopoDS_Shape:
            """
            Scales TopoDS_Shape to a given value. This is useful for scaling shapes that are in a shape with respect to another shape.
            :param item: TopoDS_Shape to be scaled.
            :param cnt_pnt: the point of the scaling center.
            :param factor: Factor to scale the shape by. Default is 1.
            :return: a scaled TopoDS_Shape with scaling applied to it.
            """
            return o.scaler(item, cnt_pnt, factor)
        
        @staticmethod
        def hollow_carve(face: TopoDS_Shape, factor: float):
            """
            (This can be replaced by cutter3D() now.)Carving on a face with a shape scaling down from itself.
            :param face: TopoDS_Shape to be cut.
            :param factor: Factor to be used to scale the cutter.
            :return: A shape with the cutter in it's center of mass scaled by factor
            """
            return o.hollow_carver(face, factor)
        
    class mesh(object):
        @staticmethod
        def gmsh_switch(s: bool) -> None:
            """
            Switch on and off the Gmsh Engine.
            :param s: True or False
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
        def write_stl(item: any, item_name: str, linear_deflection: float = 0.001, angular_deflection: float = 0.1, output_mode = 1, store_dir: str = None) -> None:
            """
            Write OCC to STL file. This function is used to write a file to the database. The file is written to a file named item_name. 
            :param item: the item to be written to the file.
            :param item_name: the name of the item. It is used to generate the file name.
            :param linear_deflection: the linear deflection factor.
            :param angular_deflection: the angular deflection factor.
            :param output_mode: for using different api in occ.
            :param store_dir the directory to store the file in.
            :return: None if success else error code ( 1 is returned if error ). In case of error it is possible to raise an exception
            """
            utw.stl_writer(item, item_name, linear_deflection, angular_deflection, output_mode, store_dir)

        @staticmethod
        def write_step(item: any, filename: str, directory: str):
            """
            Writes a step file. This is a wrapper around write_step_file to allow a user to specify the shape of the step and a filename
            :param item: the item to write to the file
            :param filename: the filename to write the file to ( default is None
            """
            utw.step_writer(item, filename, directory)
        
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
            brief Generate a name based on the type of name. It is used to generate an output name for a layer or a geometric object
            :param name_type: Type of name to generate
            :param dim_vector: Vector of dimension values ( default : None )
            :param batch_num: Number of batch to generate ( default : None )
            :param parm_title: List of parameters for the layer
            :param is_layer_thickness: True if the layer is thickness ( default : False )
            :param layer_param: Parameter of the layer ( default : None )
            :param geom_name: Name of the geometric object ( default : None )
            :return: Name of the layer or geometric object ( default : None ) - The string representation of the nam
            """
            return utw.namer(name_type, dim_vector, batch_num, parm_title, is_layer_thickness, layer_param, geom_name)
        
        @staticmethod
        def get_filename(path: str) -> str:
            return utr.get_filename(path)
        
        @staticmethod
        def read_step(path: str) -> TopoDS_Shape():
            '''
            Read STEP file to memory in OCC format
            '''
            return utr.step_reader(path)
        
        @staticmethod
        def read_stl(path: str) -> TopoDS_Shape:
            '''
            Read STL file to memory in OCC format.
            '''
            return utr.stl_reader(path)
        
        @staticmethod
        def upload(source: str, destination: str) -> bool:
            '''
            For db operation only.
            Copy one file to a specific place. 
            '''
            return dio.file_copy(path1=source, path2=destination)
        
        @staticmethod
        def get_md5(source: str) -> str:
            '''
            Get MD5 value of a file.
            '''
            return utr.get_file_md5(path=source)
        
        @staticmethod
        def mk_newdir(dirname:str, folder_name: str):
            '''
            Make a new directory with given name.
            :param dirname: Absolute path to the place for holding the new directory.
            :param folder_name: name of the new directory.
            :return: Absolute path of the new directory.
            '''
            return utw.mk_dir(dirname= dirname, folder_name=folder_name)
        
        @staticmethod
        def write_mesh(item: gmsh.model, directory: str, modelname: str, output_filename: str, format: str):
            """Writes mesh to file. This function is used to write meshes to file. The format is determined by the value of the format parameter
            :param item: gmsh. model object that contains the model
            :param directory: directory where the file is located. It is the root of the file
            :param modelname: name of the gmsh model to be written
            :param output_filename: name of the file to be written
            :param format: format of the file to be written. Valid values are vtk msh
            """
            utw.mesh_writer(item=item, directory=directory, modelname=modelname, format=format, output_filename=output_filename)
            
        @staticmethod
        def is_md5(string: str) -> bool:
            '''
            Use regex to check if a sting is MD5.
            '''
            return utr.is_md5_hash(s=string)
        
        @staticmethod
        def download(file_dir: str = None,
               output_dir: str = None,
               task_id: int | str = None,
               time_range: list = None,
               org: bool = True):
            '''
            For db operation only.
            Copy one file and translate its name from hash32 to its real name, then paste it to the destination.
            '''
            return dio.downloader(file_dir=file_dir, output_dir=output_dir, task_id=task_id, time_range=time_range, org=org)
        
        @staticmethod
        def delete(dir_path: str, filename: str = None, operate: str = None, op_list: list = None):
            '''
            For db operation only.
            Delete one or multiple files.
            '''
            dio.file_delete(dir_path=dir_path, filename=filename, operate=operate, op_list=op_list)
        
        
    
    