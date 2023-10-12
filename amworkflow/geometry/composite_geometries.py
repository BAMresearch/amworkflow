# complex geometry definitions from old composite_geometry.py file

# class CreateWallByPointsUpdate():
#     def __init__(self, coords: list, th: float, height: float):
#         self.coords = Pnt(coords).coords
#         self.height = height
#         self.R = None
#         self.interpolate = 6
#         self.th = th
#         self.is_close = True
#         self.vecs = []
#         self.central_segments = []
#         self.dir_vecs = []
#         self.ths = []
#         self.lft_coords = []
#         self.rgt_coords = []
#         self.side_coords: list
#         self.create_sides()
#         self.pnts = Segments(self.side_coords)
#         self.G = nx.from_dict_of_lists(self.pnts.pts_digraph, create_using=nx.DiGraph)
#         # self.all_loops = list(nx.simple_cycles(self.H)) # Dangerous! Ran out of memory.
#         self.loop_generator = nx.simple_cycles(self.G)
#         self.check_pnt_in_wall()
#         self.postprocessing()
#
#     def create_sides(self):