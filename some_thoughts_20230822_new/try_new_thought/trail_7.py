from amworkflow.api import amWorkflow as aw
a = [[], [2,1], [5,0], [10,9],[5,12],[3,5],[12,3]]
wall = aw.geom.CreateWallByPoints(a,2,3,False)
wall.visualize(all_polygons=False, display_central_path=True)
geom = wall.Shape()
aw.tool.write_stl(geom, "wall2", store_dir="/home/yhe/Documents/new_am2/amworkflow/some_thoughts_20230822_new/try_new_thought")