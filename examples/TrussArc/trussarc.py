import pathlib
import os
import pandas as pd
import numpy as np
from amworkflow.api import amWorkflow as aw

# python trussarc.py -n trussarc -gp thickness height -gpv 50 100 -mbt 10 -msf 5 -vtk

@aw.engine.amworkflow() #("draft") # for visulaization
def geometry_spawn(pm):
    #Define model by given file with points
    # float parameters:
    # pm.thickness: float - thickness of layers
    # pm.height: float - global height of model

    # from points list
    name = 'print110823'
    root = pathlib.Path(__file__).parent
    data = pd.read_csv(root / f"{name}.csv", sep=',')
    data['z'] = np.zeros(len(data)) # add z coordinate
    # print(data)
    pmfo = np.array(data[['x','y','z']])
    # print(len(pmfo))
    # # only for outer line
    # print(pmfo[0:90])
    g = aw.geom
    # pmfo = pmfo[0:90] # outline
    wall_maker = g.CreateWallByPoints(pmfo, pm.thickness, pm.height,is_close=False)
    design = wall_maker.Shape()
    # wall_maker.visualize # Uncomment this line if you would like to visualize it in plot.

    return design #TopoDS_Shape



