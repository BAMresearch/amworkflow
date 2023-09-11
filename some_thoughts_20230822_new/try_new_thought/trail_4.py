from amworkflow.src.geometries.composite_geometry import CreateWallByPointsUpdate
import numpy as np
from amworkflow.api import amWorkflow as aw
import pathlib
import os
import pandas as pd
import numpy as np
import time
from amworkflow.src.utils.meter import timer

name = 'print110823'
root = pathlib.Path(__file__).parent
data = pd.read_csv(root / f"{name}.csv", sep=',')
data['z'] = np.zeros(len(data)) # add z coordinate
# print(data)
pmfo = np.array(data[['x','y','z']])
print(pmfo.shape)
# print(len(pmfo))
# # only for outer line
# print(pmfo[0:90])
g = aw.geom
wall = CreateWallByPointsUpdate(pmfo, 50, 12, is_close=False)
# print(wall.loops)
# wall.visualize_graph()
wall.visualize(display_polygon=True)
poly = wall.Shape()
wall.visualize()
aw.tool.write_stl(poly, "sucess_new_scheme2",store_dir="/home/yhe/Documents/new_am2/amworkflow/some_thoughts_20230822_new/try_new_thought")

