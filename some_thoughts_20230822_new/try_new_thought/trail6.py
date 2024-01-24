from amworkflow.api import amWorkflow as aw
a = [[], [2,2], [5,0], [10,9]]
wall = aw.geom.CreateWallByPoints(a,2,3,False)
wall.visualize()
geom = wall.Shape()
aw.tool.write_stl(geom, "wall1", store_dir="/home/yhe/Documents/new_am2/amworkflow/some_thoughts_20230822_new/try_new_thought")