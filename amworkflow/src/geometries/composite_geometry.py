import copy as cp
import itertools

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Polygon
from OCC.Core.BRepFilletAPI import BRepFilletAPI_MakeFillet
from OCC.Core.gp import gp_Pnt
from OCC.Core.TopoDS import TopoDS_Face, TopoDS_Shape, TopoDS_Shell

from amworkflow.src.geometries.builder import sewer, solid_maker
from amworkflow.src.geometries.operator import (
    bender,
    cutter3D,
    fuser,
    geom_copy,
    hollow_carver,
    reverse,
    rotate_face,
    translate,
)
from amworkflow.src.geometries.property import (
    get_face_center_of_mass,
    p_bounding_box,
    p_get_face_area,
    shortest_distance_point_line,
    topo_explorer,
)
from amworkflow.src.geometries.simple_geometry import (
    Pnt,
    Segments,
    angle_of_two_arrays,
    angular_bisector,
    create_edge,
    create_face,
    create_prism,
    create_wire,
    laterality_indicator,
    linear_interpolate,
    p_center_of_mass,
    random_polygon_constructor,
)
from amworkflow.src.utils.meter import timer


def polygon_maker(side_num: int,
                  side_len: float,
                  rotate: float = None,
                  bound: bool = False):
    """
    @brief Creates a regular polygon. The polygon is oriented counterclockwise around the origin. If bound is True the polygon will be only the boundary (TopoDS_Wire) the polygon.
    @param side_num Number of sides of the polygon.
    @param side_len Length of the side of the polygon.
    @param rotate Rotation angle ( in radians ). Defaults to None which means no rotation.
    @param bound output only the boundary. Defaults to False. See documentation for create_wire for more information.
    @return face or boundary of the polygon.
    """
    vertices = []
    sides = []
    r = side_len / (2 * np.sin(np.pi/side_num))
    ang = np.linspace(0,2*np.pi, side_num + 1)
    if rotate != None:
        ang += rotate
    for i in range(side_num + 1):
        if i != side_num:
            x = np.sin(ang[i]) * r
            y = np.cos(ang[i]) * r
            t_pnt = gp_Pnt(x, y, 0)
            vertices.append(t_pnt)
        if i != 0:
            if i == side_num :
                t_edge = create_edge(pnt1=vertices[0], pnt2=vertices[-1])
                sides.append(t_edge)
                wire = create_wire(wire, t_edge)
            elif i == 1:
                t_edge = create_edge(pnt1=vertices[i-1], pnt2=t_pnt)
                sides.append(t_edge)
                wire = create_wire(t_edge)
            else:
                t_edge = create_edge(pnt1=vertices[i-1], pnt2=t_pnt)
                sides.append(t_edge)
                wire = create_wire(wire, t_edge)
    
    face = create_face(wire)
    if bound:
        return wire
    else:
        return reverse(face)

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

    def multiplier_unit(face_odd, face_even, unit_len):
        """
         @brief Fuse a unit of mass to an odd or even side number.
         @param face_odd face with odd side number ( numpy array )
         @param face_even face with even side number ( numpy array )
         @param unit_len length of side ( numpy array )
         @return one unit of hexagon multiplication.
        """
        if side_num % 2 == 0:
            face = face_even
            ang = np.linspace(0,2*np.pi, side_num + 1)
        else:
            face = face_odd
            ang = np.linspace(np.pi / side_num,2*np.pi + np.pi / side_num, side_num + 1)
        cnt = get_face_center_of_mass(face)
        len = unit_len * 0.5 / np.tan(np.pi / side_num) * 2
        face_collect = [face]
        fuse = face
        for ag in ang:
            rot_face = rotate_face(face, ag)
            x = len * np.sin(ag)
            y = len * np.cos(ag)
            translate(rot_face, [cnt[0] + x, cnt[1] + y, 0])
            face_collect.append(rot_face)
            fuse = fuser(fuse, rot_face)
        return fuse
    # This function will generate a hollow carver for each iteration.
    for i in range(iter_num):
        # This function computes the fuse of the carver.
        if i == 0:
            face_even = hollow_carver(polygon_maker(side_num, side_len, rotate=-np.pi / side_num), wall)
            face_odd = hollow_carver(polygon_maker(side_num, side_len),wall)
            fuse = multiplier_unit(face_odd, face_even, unit_len=side_len)
        else:
            face_even = fuse
            # face_odd = rotate_face(fuse,angle = -np.pi / side_num) 
            face_odd = fuse
            fuse = multiplier_unit(face_odd, face_even, unit_len=3 * i * side_len)
    # translate center to use list of coordinates
    if isinstance(center, gp_Pnt):
        translate(fuse, list(center.Coord()))
    return fuse

def isoceles_triangle_maker(bbox_len:float, bbox_wid: float, thickness: float = None):
    """
     @brief (Having problem with wall thickness now.) Create isoceles triangulation. This is a function to create isoceles triangulation of a bounding box and its widest corner
     @param bbox_len length of bounding box of the triangle
     @param bbox_wid width of bounding box of the triangle
     @param thickness thickness of the wall of the triangle
     @return a hollowed triangle face.
    """
    h = bbox_wid
    l = bbox_len
    t = thickness if thickness != None else 0
    ary = np.array([h,l])
    r1 = np.linalg.norm(ary*0.5, 2)
    ang1 = np.arccos((h*0.5) / r1)
    ang2 = np.pi - ang1
    r = [h*0.5, r1, r1]
    thet = np.arctan(l * 0.5 / h)
    thet1 = np.arctan(h / l)
    t1 = t / np.sin(thet)
    t2 = t / np.sin(thet1)
    h1 = h - t - t1
    l1 = h1 * l / h
    ary1 = np.array([h1, l1])
    r2 = np.linalg.norm(0.5 * ary1, 2)
    ri = [0.5 * h1, r2, r2]
    ang = [0, ang2, -ang2]
    def worker(r):
        """
         @brief Creates a triangle in a polarized coordinate system.
         @param r The radius of each vertex.
         @return a triangle
        """
        pt_lst = []
        # Add a point on the plane.
        for i in range(3):
            x = r[i] * np.sin(ang[i])
            y = r[i] * np.cos(ang[i])
            pnt = gp_Pnt(x,y,0)
            pt_lst.append(pnt)
        # Create a wire for each point in the list of points.
        for i in range(3):
            # Create a wire for the i th point of the point.
            if i == 0:
                edge = create_edge(pt_lst[i], pt_lst[i+1])
                wire = create_wire(edge)
            elif i != 2:
                edge = create_edge(pt_lst[i], pt_lst[i+1])
                wire = create_wire(wire, edge)
            else:
                edge = create_edge(pt_lst[i], pt_lst[0])
                wire = create_wire(wire, edge)
        face = create_face(wire)
        return reverse(face)
    outer_face = worker(r)
    # The face to be cuttered.
    if t != 0:
        inner_face = worker(ri)
        new_face = cutter3D(outer_face, inner_face)
        return new_face
    else:
        return outer_face
    
def create_sym_hexagon1_infill(total_len: float, total_wid:float, height:float, th: float) :
    """
     @brief Create an infill pattern using symmetrical hexagon with defined len, height and numbers.
     @param total_len total length of the bounding box.
     @param total_wid total wid of the bounding box.
     @param height height of the prism. This is the same as height of the hexagon.
     @param th thickness of the wall of the hexagon.
     @return 
    """
    p0 = [0,th * 0.5]
    p1 = []

def find_intersect(lines: np.ndarray) -> np.ndarray:
    parallel = False
    coplanarity = False
    l1, l2 = lines
    pt1, pt2 = l1
    pt3, pt4 = l2
    L1 = (pt2 - pt1) / np.linalg.norm(pt2 - pt1)
    L2 = (pt4 - pt3) / np.linalg.norm(pt4 - pt3)
    V1 = pt4 - pt1
    D1 = np.linalg.norm(V1)
    if np.isclose(np.dot(L1, L2),0) or np.isclose(np.dot(L1, L2),-1):
        parallel = True
        print("Two lines are parallel.")
        return np.full((3,1), np.nan)
    indicate = np.linalg.det(np.array([V1, L1, L2]))
    if np.abs(indicate) < 1e-8:
        coplanarity = True
    else:
        print("lines are not in the same plane.")
        return np.full((1,3), np.nan)
    if coplanarity and not parallel:
        if np.isclose(D1,0):
            return pt1
        else:
            pt5_pt4 = np.linalg.norm(np.cross(V1, L1))
            theta = np.arccos(np.dot(L1, L2))
            o_pt5 = pt5_pt4 / np.tan(theta)
            o_pt4 = pt5_pt4 / np.sin(theta)
            V1_n = V1 / D1
            cos_beta = np.dot(V1_n, L1)
            pt1_pt5 = D1 * cos_beta
            pt1_o = pt1_pt5 - o_pt5
            o = L1 * pt1_o + pt1
            return o

def index2array(ind: list, array: np.ndarray):
    real_array = []
    for i,v in enumerate(ind):
        item = []
        for j, vv in enumerate(v):
            item.append(array[vv])
        real_array.append(item)
    return real_array

def polygon_interpolater(plg: np.ndarray, step_len: float = None, num: int = None, isclose: bool = True):
    def deter_dum(line: np.ndarray):
        ratio = step_len / np.linalg.norm(line[0]-line[1])
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
        elif ratio <=0.14:
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
        line = np.array([pt, plg[i+1]])
        if num is not None:
            p_num = num
        else:
            p_num = deter_dum(line)
        insert_p = linear_interpolate(line, p_num)
        new_plg = np.concatenate((new_plg[:pos+1],insert_p, new_plg[pos+1:]))
        pos +=p_num+1
    return new_plg

# @timer
class CreateWallByPointsUpdate():
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
        self.side_coords: list
        self.create_sides()
        self.pnts = Segments(self.side_coords)
        self.G = nx.from_dict_of_lists(self.pnts.pts_digraph, create_using=nx.DiGraph)
        # self.all_loops = list(nx.simple_cycles(self.H)) # Dangerous! Ran out of memory.
        self.loop_generator = nx.simple_cycles(self.G)
        self.check_pnt_in_wall()
        self.postprocessing()
        
    def create_sides(self):
        if self.R is not None:
            self.coords = polygon_interpolater(self.coords, self.interpolate)
            self.coords = bender(self.coords, self.R)
            self.coords = [i for i in self.coords]
        self.th *= 0.5
        for i,p in enumerate(self.coords):
            if i != len(self.coords) - 1:
                self.central_segments.append([self.coords[i], self.coords[i+1]])
                a1 = self.coords[i+1] - self.coords[i]
                if i == 0:
                    if self.is_close:
                        dr = angular_bisector(self.coords[-1] - p, a1)
                        # ang = angle_of_two_arrays(dir_vecs[i-1],dr)
                        ang2 = angle_of_two_arrays(laterality_indicator(p - self.coords[-1], True), dr)
                        ang_th = ang2
                        if ang2 > np.pi / 2:
                            dr *= -1
                            ang_th = np.pi - ang2
                        nth = np.abs(self.th / np.cos(ang_th))
                    else:
                        dr = laterality_indicator(a1, True)
                        nth = self.th
                else:
                    dr = angular_bisector(-self.vecs[i-1], a1)
                    ang2 = angle_of_two_arrays(laterality_indicator(self.vecs[i-1], True), dr)
                    ang_th = ang2
                    if ang2 > np.pi / 2:
                        dr *= -1
                        ang_th = np.pi - ang2
                    nth = np.abs(self.th / np.cos(ang_th))
            else:
                if self.is_close:
                    self.central_segments.append([self.coords[i], self.coords[0]])
                    a1 = self.coords[0] - self.coords[i]
                    dr = angular_bisector(-self.vecs[i-1], a1)
                    ang2 = angle_of_two_arrays(laterality_indicator(self.vecs[i-1], True), dr)
                    ang_th = ang2
                    if ang2 > np.pi / 2:
                        dr *= -1
                        ang_th = np.pi - ang2 
                    nth = np.abs(self.th / np.cos(ang_th))
                else:
                    dr = laterality_indicator(a1, True)
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
                lmbda, dist = shortest_distance_point_line(vec,coord)
                if dist < 0.9 * self.th:
                    print(f"pnt:{pnt},dist:{dist},lmbda:{lmbda}, vec:{vec}")
                    self.in_wall_pts_list.update({pnt:True})
                    break

        
    def visualize(self, display_polygon: bool = True,display_central_path:bool = False,all_polygons:bool = False):
        # Extract the x and y coordinates and IDs
        a = self.pnts.pts_index
        x = [coord[0] for coord in a.values()]
        y = [coord[1] for coord in a.values()]
        ids = list(a.keys())  # Get the point IDs
        # Create a scatter plot in 2D
        plt.subplot(1,2,1)
        # plt.figure()
        plt.scatter(x, y)

        # Annotate points with IDs
        for i, (xi, yi) in enumerate(zip(x, y)):
            plt.annotate(f'{ids[i]}', (xi, yi), fontsize=12, ha='right')
        
        if display_polygon:
            if all_polygons:
                display_loops = nx.simple_cycles(self.G)
            else:
                display_loops = self.result_loops
            for lp in display_loops:              
                coords = [self.pnts.pts_index[i] for i in lp]
                x = [point[0] for point in coords]
                y = [point[1] for point in coords]
                plt.plot(x + [x[0]], y + [y[0]], linestyle='-', marker='o')
        if display_central_path:
            talist = np.array(self.coords).T
            x1 = talist[0]
            y1 = talist[1]
            plt.plot(x1, y1, 'bo-', label='central path', color = "b")
        
        # Create segments by connecting consecutive points

        # a_subtitute = np.array(self.side_coords)
        # toutput1 = a_subtitute.T
        # x2 = toutput1[0]
        # y2 = toutput1[1]
        # plt.plot(x2, y2, 'ro-', label='outer line')
            
        # Set labels and title
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('Points With Polygons detected')

        plt.subplot(1,2,2)
        # layout = nx.spring_layout(self.G)
        layout = nx.circular_layout(self.G)
        # Draw the nodes and edges
        nx.draw(self.G, pos=layout, with_labels=True, node_color='skyblue', font_size=10, node_size=300)
        plt.title("Multi-Digraph")
        plt.tight_layout() 
        # Show the plot
        plt.show()
    
    def get_loops(self):
        return [i for i in self.all_loops if len(i) > 2]

    def visualize_graph(self):
        layout = nx.spring_layout(self.G)
        # Draw the nodes and edges
        nx.draw(self.G, pos=layout, with_labels=True, node_color='skyblue', font_size=10, node_size=500)
        plt.title("NetworkX Graph Visualization")
        plt.show()
        
    def Shape(self):
        loop_r = self.rank_result_loops()
        print(loop_r)
        boundary = [self.occ_pnt(self.pnts.pts_index[i]) for i in loop_r[0]]
        poly0 = random_polygon_constructor(boundary)
        poly_r = poly0
        for i, h in enumerate(loop_r):
            if i == 0:
                continue
            h = [self.occ_pnt(self.pnts.pts_index[i]) for i in h]
            poly_c = random_polygon_constructor(h)
            poly_r = cutter3D(poly_r, poly_c)
        self.poly = poly_r
        if not np.isclose(self.height, 0):
            wall_compound = create_prism(self.poly, [0,0,self.height])
            faces = topo_explorer(wall_compound, "face")
            wall_shell = sewer(faces)
            self.wall = solid_maker(wall_shell)
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
                for i,pt in enumerate(lp):
                    if i == 0:
                        if pt in self.in_wall_pts_list:
                            in_wall_pt_count += 1
                        if (lp[-1],pt) in self.pnts.virtual_vector:
                            virtual_vector_count += 1
                    else:
                        if pt in self.in_wall_pts_list:
                            in_wall_pt_count += 1
                        if (lp[i-1],pt) in self.pnts.virtual_vector:
                            virtual_vector_count += 1
                    if (in_wall_pt_count > 0) or (virtual_vector_count > 1) or ((in_wall_pt_count == 0) and (virtual_vector_count > 0)):
                    # if (in_wall_pt_count > 0):
                        visible_loop = False
                        break  
            else:
                real_loop = False           
            if real_loop and visible_loop:
                self.result_loops.append(lp)
                for pt in lp:
                    if pt not in self.in_loop_pts_list:
                        self.in_loop_pts_list.update({pt:[correct_loop_count]})
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
        self.result_loops = [v for i,v in enumerate(self.result_loops) if i not in filtered_lp]
        print("result:", self.result_loops)

    def rank_result_loops(self):
        areas = np.zeros(len(self.result_loops))
        for i,lp in enumerate(self.result_loops):
            lp_coord = [self.pnts.pts_index[i] for i in lp]
            area = p_get_face_area(lp_coord)
            areas[i] = area
        rank = np.argsort(areas).tolist()
        self.result_loops = sorted(self.result_loops, key=lambda x: rank.index(self.result_loops.index(x)),reverse=True)
        return self.result_loops
        
    def occ_pnt(self,coord) -> gp_Pnt:
        return gp_Pnt(*coord)