from amworkflow.src.geometries.composite_geometry import CreateWallByPointsUpdate
import numpy as np
from amworkflow.api import amWorkflow as aw
import pathlib
import os
import pandas as pd
import numpy as np

name = 'print110823'
root = pathlib.Path(__file__).parent
data = pd.read_csv(root / f"{name}.csv", sep=',')
data['z'] = np.zeros(len(data)) # add z coordinate
# print(data)
pmfo = np.array(data[['x','y','z']])
print(pmfo)
# print(len(pmfo))
# # only for outer line
# print(pmfo[0:90])
g = aw.geom
wall = CreateWallByPointsUpdate(pmfo, 8, 12)
# print(wall.loops)
wall.visualize_graph()
wall.visualize()
