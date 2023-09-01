import numpy as np
import matplotlib.pyplot as plt

class Pnt():
    def __init__(self, *coords: list):
        self.coords = coords
        self.pts_numpy: np.ndarray
        if (len(self.coords) == 1) and (type(self.coords[0]) is list):
            self.coords = coords[0]
        self.create()
    
    def enclose(self):
        distance = np.linalg.norm(self.pts_numpy[-1] - self.pts_numpy[0])
        if np.isclose(distance,0):
            print("Polygon seems already enclosed, skipping...")
        else:
            self.pts_numpy = np.vstack((self.pts_numpy, self.pts_numpy[0]))
            self.create_attr()
        
    def create(self):
        self.pts_numpy = self.create_pnts()
        self.create_attr()
        
    def create_attr(self):
        self.pts_to_list = self.pts_numpy.tolist()
        self.pts_to_gp_Pnt: list
        self.x, self.y, self.z = self.pts_numpy.T
        self.pts_num = self.pts_numpy.shape[0]
    
    def pnt(self,pt_coord) -> np.ndarray:
        opt = np.array(pt_coord)
        dim = len(pt_coord)
        if dim > 3:
            raise Exception(f"Got wrong point {pt_coord}: Dimension more than 3rd provided.")
        if dim < 3:
            opt = np.lib.pad(opt, ((0,3 - dim)), "constant", constant_values=0)
        return opt

    def create_pnts(self) -> np.ndarray:
        for i, pt in enumerate(self.coords):
            if i == 0:
                pts = np.array([self.pnt(pt)])
            else:
                pts = np.vstack([pts, self.pnt(pt)])
        return pts

class Segments(Pnt):
    def __init__(self,*coords: list):
        super().__init__(*coords)
        self.enrch_pts = self.init_pnts()
        self.count_pts_id = self.pts_num - 1
        self.count_vector_id = self.pts_num - 1
        
    def init_pnts(self):
        enrch_pts = np.zeros((self.pts_num,5))
        for i in range(self.pts_num):
            enrch_pts[i] = np.concatenate((self.pts_numpy[i],(i,1)))
        return enrch_pts
    
    def enclose(self):
        pass
    
    def segment(self):
        pass
    
    def new_pnt(self, pt_coords: list, num_of_vector: int):
        pt_coords = np.array(pt_coords)
        opt_pts = np.concatenate((pt_coords, (self.count_pts_id+1, num_of_vector)))
        self.count_pts_id += 2
        return opt_pts
    
    def insert_item(self, *items: np.ndarray, original: np.ndarray, insert_after: int) -> np.ndarray:
        print(original[:insert_after+1])
        return np.concatenate((original[:insert_after+1], items, original[insert_after+1:]))
    
    
    
class CheckSegments():
    def __init__(self, s1: np.ndarray, s2: np.ndarray) -> None:
        self.s1 = s1
        self.s2 = s2
        self.L1, self.st1, self.et1 = self.s1
        self.L2, self.st2, self.et2 = self.s2
        self.L3 = self.et2 - self.st1
        self.L4 = self.st1 - self.et2
        self.L5 = self.st2 - self.st1
        self.L6 = self.et2 - self.et1
        self.D1 = np.linalg.norm(self.et1 - self.st1)
        self.D2 = np.linalg.norm(self.et2 - self.st2)
        self.D3 = np.linalg.norm(self.et2-self.st1)
        self.D4 = np.linalg.norm(self.st2-self.et1)
        self.D5 = np.linalg.norm(self.st2-self.st1)
        self.D6 = np.linalg.norm(self.et2 - self.et1)
        self.N1 = self.get_normalize_vector(self.L1)
        self.N2 = self.get_normalize_vector(self.L2)
        self.are_parallel = False
        self.are_colinear = False
        self.are_coplanar = False
        self.intersect = None
        self.on_s1 = False
        self.on_s2 = False
        
    def check_relationship(self):
        coplanarity = np.linalg.det(np.array([self.L3, self.L1, self.L2]))
        if np.isclose(coplanarity, 0):
            self.are_coplanar = True
        else:
            pass
        parallel = np.linalg.norm(self.L1 - self.L2)
        if np.isclose(parallel, 0):
            self.are_parallel = True
    
    def check_exceptions(self):
        if np.isclose(self.D1+self.D4-self.D5, 0) and np.isclose(self.D1+self.D6-self.D3, 0):
            self.are_colinear = True
        
    
    def get_normalize_vector(self, v: np.ndarray):
        return v / np.linalg.norm(v)
    
    
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



pts = Pnt([0,0,0], [0, 5, 0], [5,5,0], [2,2], [-2,2])
segments = Segments([0,0,0], [0, 5, 0], [5,5,0], [2,2], [-2,2])

pts.enclose()
x = pts.x
y = pts.y


def plot_intersect(x11, x12, y11, y12, x21, x22, y21, y22):
    # Coordinates for the two segments
    segment1_x = [x11, x12]
    segment1_y = [y11, y12]

    segment2_x = [x21, x22]
    segment2_y = [y21, y22]

    intersect = find_intersect(np.array([[[x11, y11,0], [x12, y12,0]], [[x21, y21,0], [x22, y22,0]]]))
    # Coordinates for the single point
    point_x = [intersect[0]]
    point_y = [intersect[1]]

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the two segments
    ax.plot(segment1_x, segment1_y, color='blue', linestyle='-', linewidth=2, label='Segment 1')
    ax.plot(segment2_x, segment2_y, color='green', linestyle='-', linewidth=2, label='Segment 2')

    # Plot the single point
    ax.plot(point_x, point_y, marker='o', markersize=8, color='red', label='Point')

    # Add labels for the point and segments
    ax.text(2, 3, f'Point ({intersect[0]}, {intersect[1]})', fontsize=12, ha='right')
    ax.text(1, 2, 'Segment 1', fontsize=12, ha='right')
    ax.text(6, 3, 'Segment 2', fontsize=12, ha='right')

    # Add a legend
    ax.legend()

    # Set axis limits for better visualization
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 6)

    # Set axis labels
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')

    # Set plot title
    ax.set_title('Two Segments and One Point')

    # Display the plot
    plt.show()
    
plot_intersect(2,5,3,5,5,2,5,3)
    

def plot_segments(x,y):
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the points as red dots
    ax.plot(x, y, marker='o', color='red', linestyle='-', markersize=8, label='Points')

    # Plot the segments between the points as a blue line
    ax.plot(x, y, color='blue', linestyle='-', linewidth=1, label='Segments')

    # Add labels to the points
    for i, (xi, yi) in enumerate(zip(x, y)):
        ax.text(xi, yi, f'{i}, ({xi}, {yi})', fontsize=12, ha='right')

    # Add arrows to indicate the direction of segments
    for i in range(len(x) - 1):
        ax.annotate('', xy=(x[i+1], y[i+1]), xytext=(x[i], y[i]),
                    arrowprops=dict(arrowstyle='->', color='green', linewidth=1.5),
                    annotation_clip=False)

    # Add a legend
    ax.legend()

    # Set axis labels
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')

    # Set plot title
    ax.set_title('2D Points and Segments with Arrows')

    # Display the plot
    plt.show()