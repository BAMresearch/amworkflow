from amworkflow.src.geometries.composite_geometry import CreateWallByPointsUpdate
import numpy as np
from amworkflow.api import amWorkflow as aw

points = [[],[0,2],[2,4],[5,4],[8,4],[8,6],[3,6],[3,1]]
wall = CreateWallByPointsUpdate(points,1,2)
print(wall.loops)
wall.visualize_graph()
wall.visualize()