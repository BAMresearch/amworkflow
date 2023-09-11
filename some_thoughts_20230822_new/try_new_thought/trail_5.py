from amworkflow.src.geometries.composite_geometry import CreateWallByPointsUpdate
import numpy as np
from amworkflow.api import amWorkflow as aw

points = [[],[0,2],[2,4],[5,4],[8,4],[8,6],[3,6],[3,1]]
wall = CreateWallByPointsUpdate(points,1,2)
# wall.visualize_graph()
poly = wall.Shape()

aw.tool.write_stl(poly, "sucess_new_scheme3",store_dir="/home/yhe/Documents/new_am2/amworkflow/some_thoughts_20230822_new/try_new_thought")
wall.visualize()