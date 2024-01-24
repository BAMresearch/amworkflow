from amworkflow.src.geometries.simple_geometry import Segments
from amworkflow.src.geometries.property import check_parallel_line_line, check_overlap, shortest_distance_line_line
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from amworkflow.src.utils.visualizer import plot_digraph

sgmts = Segments([0,0,0], [0, 5, 0], [5,5,0], [2,2], [-2,2],[0,5],[0,6], [])
print(sgmts.pts_index)
print(sgmts.pts_digraph)
print(sgmts.init_pts_sequence)
print(sgmts.modify_edge_list)
sgmts.modify_edge()
print(sgmts.pts_digraph)

